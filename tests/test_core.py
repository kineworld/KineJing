import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from kinejing.adapters import plan,execute
from kinejing.features import rank
from kinejing.refinement import flow_knots,render
from kinejing.motion import align


class CoreTests(unittest.TestCase):
    def config(self):
        return dict(backend='causalwm',repo='external/repo',image='image.png',prompt='a; $(echo nope)',checkpoint='weights',base_checkpoint='base',text_encoder='text',output='new-output')

    def test_prompt_is_one_argv_element(self):
        c=self.config();job=plan(c)
        self.assertEqual(job['argv'][job['argv'].index('--prompt')+1],c['prompt'])

    def test_unsupported_actions_fail(self):
        c=self.config();c['actions']='robot.npy'
        with self.assertRaises(ValueError):plan(c)

    def test_bad_temporal_shape_fails(self):
        c=self.config();c['frames']=120
        with self.assertRaises(ValueError):plan(c)

    def test_all_example_plans(self):
        root=Path(__file__).resolve().parents[1]
        for name in ['causalwm','wan22','vjepa21','kine-jepa']:
            config=json.loads((root/'examples'/(name+'.json')).read_text())
            job=plan(config)
            self.assertEqual(job['backend'],name)
            self.assertEqual(len(job['expected_revision']),40)
        wan=plan(json.loads((root/'examples/wan22.json').read_text()))
        self.assertIn('ti2v-5B',wan['argv'])
        self.assertIn('--base_seed',wan['argv'])

    def test_missing_weights_never_launch(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config();c['repo']=d
            with patch('kinejing.adapters.subprocess.run') as run:
                with self.assertRaises(FileNotFoundError):execute(c)
                run.assert_not_called()

    def test_wrong_revision_never_launch(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config();c['repo']=d
            for key in ['image','checkpoint','base_checkpoint','text_encoder']:
                c[key]=str(Path(d)/key);Path(c[key]).write_text('fixture')
            (Path(d)/'inference.py').write_text('')
            with patch('kinejing.adapters.subprocess.check_output',return_value='different'),patch('kinejing.adapters.subprocess.run') as run:
                with self.assertRaises(ValueError):execute(c)
                run.assert_not_called()

    def test_bounded_motion_and_first_frame(self):
        rng=np.random.RandomState(42);image=rng.randint(0,256,(240,320,3)).astype(np.uint8)
        donor=np.stack([image,np.roll(image,12,axis=1)])
        rigid=flow_knots(image,donor,0);bounded=flow_knots(image,donor,.25)
        self.assertLessEqual(float(np.linalg.norm(bounded-rigid,axis=-1).max()),2.00001)
        np.testing.assert_array_equal(render(image,bounded)[0],image)

    def test_alignment_monotone_and_input_only(self):
        q=np.repeat(np.linspace(0,1,17)[:,None],14,axis=1)
        path=align({'qs':q},{'q':q})
        self.assertEqual(path[0],0);self.assertTrue((np.diff(path)>=0).all())
        np.testing.assert_array_equal(path,np.arange(17))

    def test_feature_ranking_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            def save(name,x,sha='same'):
                p=Path(d)/(name+'.npz');np.savez(p,embedding=x,metadata=np.array(json.dumps({'encoder':'fixture','checkpoint_sha256':sha})));return p
            goal=save('goal',[1,0]);good=save('good',[1,0]);bad=save('bad',[0,1])
            self.assertEqual(rank(goal,[bad,good])['results'][0]['candidate'],str(good))
            incompatible=save('other',[1,0],'other')
            with self.assertRaises(ValueError):rank(goal,[incompatible])

    def test_invalid_features_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.npz';np.savez(p,embedding=[float('nan')],metadata=np.array('{}'))
            with self.assertRaises(ValueError):rank(p,[p])

if __name__=='__main__':unittest.main()
