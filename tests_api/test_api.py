import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from fastapi.testclient import TestClient
from kinejing.api import create_app

KEY='unit-test-key-not-a-real-secret'


class FakeEngine:
    def __init__(self):self.calls=0
    def predict(self,request):
        self.calls+=1
        return {'output_kind':'test_fixture_not_model_output','shape':[9,3,768]}


class ApiTests(unittest.TestCase):
    def setUp(self):self.engine=FakeEngine()
    def body(self):return {'action_schema':'triworldbench-joint-action-vector-v1','images':dict(head='a',left='b',right='c'),'actions':[[0.]*14 for _ in range(9)]}
    def test_requires_key_and_does_not_infer_without_auth(self):
        with self.assertRaises(ValueError):create_app(api_key='short')
        with TestClient(create_app(lambda:self.engine,KEY)) as c:
            self.assertEqual(c.get('/health').status_code,200)
            self.assertEqual(c.post('/v1/predictions',json=self.body()).status_code,401)
            self.assertEqual(self.engine.calls,0)
    def test_valid_request_and_model_disclosure(self):
        with TestClient(create_app(lambda:self.engine,KEY)) as c:
            headers={'Authorization':'Bearer '+KEY}
            self.assertIn('HTTPBearer',c.get('/openapi.json').json()['components']['securitySchemes'])
            self.assertFalse(c.get('/v1/models',headers=headers).json()['models'][0]['commercial_ready'])
            r=c.post('/v1/predictions',json=self.body(),headers=headers)
            self.assertEqual(r.status_code,200);self.assertIn('request_id',r.json())
            self.assertEqual(self.engine.calls,1)
    def test_bad_schema_shape_and_oversized_body_rejected(self):
        with TestClient(create_app(lambda:self.engine,KEY)) as c:
            h={'Authorization':'Bearer '+KEY}
            body=self.body();body['action_schema']='unknown'
            self.assertEqual(c.post('/v1/predictions',json=body,headers=h).status_code,422)
            body=self.body();body['actions'][0]=[0.]
            self.assertEqual(c.post('/v1/predictions',json=body,headers=h).status_code,422)
            self.assertEqual(c.post('/v1/predictions',content=b'x'*(4*1024*1024+1),headers=h).status_code,413)
            self.assertEqual(self.engine.calls,0)

    def test_busy_model_rejected_and_failure_releases_lock(self):
        entered=threading.Event();release=threading.Event()
        class BlockingEngine:
            def predict(self,request):
                entered.set();release.wait(timeout=5)
                raise RuntimeError('private internal detail must not leak')
        with TestClient(create_app(BlockingEngine,KEY)) as c, ThreadPoolExecutor(max_workers=1) as pool:
            h={'Authorization':'Bearer '+KEY}
            pending=pool.submit(c.post,'/v1/predictions',json=self.body(),headers=h)
            self.assertTrue(entered.wait(timeout=5))
            try:self.assertEqual(c.post('/v1/predictions',json=self.body(),headers=h).status_code,429)
            finally:release.set()
            result=pending.result(timeout=5)
            self.assertEqual(result.status_code,503)
            self.assertNotIn('private internal detail',result.text)
            # A new request reaches the engine after the failed request releases its lock.
            self.assertEqual(c.post('/v1/predictions',json=self.body(),headers=h).status_code,503)

if __name__=='__main__':unittest.main()
