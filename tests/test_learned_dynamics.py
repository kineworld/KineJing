import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional neural environment required')
class DynamicsTests(unittest.TestCase):
    def test_action_prefix_cannot_read_future(self):
        from kinejing.learned_dynamics import TriViewDynamics
        torch.manual_seed(3)
        m=TriViewDynamics(feature_dim=4, action_dim=2, hidden=16).eval()
        torch.nn.init.normal_(m.head[-1].weight)
        x=torch.randn(2,3,4); a=torch.randn(2,5,2)
        t=torch.linspace(0,1,5)[None,:,None].expand(2,-1,-1)
        before=m(x,a,t)
        changed=a.clone();changed[:,3:]+=10
        after=m(x,changed,t)
        torch.testing.assert_close(before[:,:3],after[:,:3],rtol=0,atol=0)
        torch.testing.assert_close(before[:,0],x,rtol=0,atol=0)
        self.assertGreater((before[:,3:]-after[:,3:]).abs().max().item(),0)

    def test_checkpoint_roundtrip(self):
        from kinejing.learned_dynamics import TriViewDynamics
        m=TriViewDynamics(feature_dim=4,action_dim=2,hidden=16)
        copy=TriViewDynamics(**m.config);copy.load_state_dict(m.state_dict(),strict=True)
        for k,v in m.state_dict().items():
            torch.testing.assert_close(v,copy.state_dict()[k])


if __name__=='__main__':unittest.main()
