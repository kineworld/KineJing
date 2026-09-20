import json,tempfile,unittest
from pathlib import Path
import numpy as np
from kinejing.rollout_io import pack,validate_inputs

class RolloutIOTests(unittest.TestCase):
    def setUp(self):
        self.tokens=np.ones((1,3,4));self.actions=np.zeros((1,2,2))
        self.meta={'encoder':'synthetic-test-only','checkpoint_sha256':'fixture'}
        self.schema={'columns':['x','y'],'units':['m','m'],'coordinate_frame':'robot_base','normalization':'none'}
        self.state={'config':{'dim':4,'action_dim':2},'encoder_provenance':self.meta,'action_schema':self.schema}

    def test_wrong_units_and_missing_schema_rejected(self):
        for schema in [{**self.schema,'units':['mm','mm']}, {**self.schema,'normalization':'z-score'}]:
            with self.assertRaises(ValueError):validate_inputs(self.tokens,self.actions,self.meta,schema,self.state)
        state=dict(self.state);state.pop('action_schema')
        with self.assertRaises(ValueError):validate_inputs(self.tokens,self.actions,self.meta,self.schema,state)

    def test_empty_and_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):validate_inputs(self.tokens[:,:0],self.actions,self.meta,self.schema)
        self.actions[0,0,0]=np.nan
        with self.assertRaises(ValueError):validate_inputs(self.tokens,self.actions,self.meta,self.schema)

    def test_pack_preserves_values_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);np.savez(p/'features.npz',tokens=self.tokens,metadata=np.array(json.dumps(self.meta)))
            np.save(p/'actions.npy',self.actions);(p/'schema.json').write_text(json.dumps(self.schema))
            args=[p/'features.npz',p/'actions.npy',p/'schema.json',p/'result.npz']
            pack(*args)
            with np.load(args[-1],allow_pickle=False) as x:
                np.testing.assert_array_equal(x['tokens'],self.tokens);np.testing.assert_array_equal(x['actions'],self.actions)
            with self.assertRaises(FileExistsError):pack(*args)

if __name__=='__main__':unittest.main()
