import json
import unittest
from backend.app import app, db
from backend.models import Settings

class TestVisionBridgeAPI(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        
        # Initialize memory database
        with app.app_context():
            db.session.remove()
            db.engine.dispose()
            db.create_all()
            # Add initial settings
            default_settings = Settings(
                mode="standard",
                confidence_threshold=0.45,
                tts_rate=160,
                tts_volume=1.0,
                night_mode=False
            )
            db.session.add(default_settings)
            db.session.commit()

    def tearDown(self):
        with app.app_context():
            db.drop_all()

    def test_get_settings(self):
        """Test GET /api/settings"""
        response = self.client.get('/api/settings')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['mode'], 'standard')
        self.assertEqual(data['confidence_threshold'], 0.45)

    def test_post_settings(self):
        """Test POST /api/settings"""
        payload = {
            "mode": "verbose",
            "confidence_threshold": 0.65,
            "tts_rate": 180,
            "tts_volume": 0.8,
            "night_mode": True
        }
        response = self.client.post(
            '/api/settings',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['mode'], 'verbose')
        self.assertEqual(data['confidence_threshold'], 0.65)
        self.assertEqual(data['tts_rate'], 180)
        self.assertTrue(data['night_mode'])

    def test_session_status_inactive(self):
        """Test GET /api/session/status when inactive"""
        response = self.client.get('/api/session/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['active'])

    def test_session_start_mock(self):
        """Test starting a session with a mock source"""
        payload = {
            "source": "mock"
        }
        response = self.client.post(
            '/api/session/start',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('session_id', data)
        self.assertEqual(data['status'], 'started')

        # Check status shows active
        status_resp = self.client.get('/api/session/status')
        status_data = json.loads(status_resp.data)
        self.assertTrue(status_data['active'])
        self.assertEqual(status_data['session_id'], data['session_id'])

        # Stop session
        stop_resp = self.client.post('/api/session/stop')
        self.assertEqual(stop_resp.status_code, 200)
        stop_data = json.loads(stop_resp.data)
        self.assertEqual(stop_data['session_id'], data['session_id'])

    def test_list_known_faces(self):
        """Test GET /api/known_faces/list"""
        response = self.client.get('/api/known_faces/list')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)

if __name__ == '__main__':
    unittest.main()
