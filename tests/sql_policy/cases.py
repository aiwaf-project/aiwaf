import json


class FrameworkCases:
    def test_both_confirmed_bypasses_blocked(self):
        for email in ["admin@juice-sh.op' OR 1=1--", "admin@juice-sh.op'--"]:
            with self.subTest(email=email):
                status, body = self.send({'email': email, 'password': 'x'})
                self.assertEqual(status, 403)
                self.assertTrue(body['error'].startswith('sql_'))

    def test_ordinary_body_reaches_handler_unchanged(self):
        payload = {'email': "o'reilly@example.com", 'password': 'ordinary#123'}
        status, body = self.send(payload)
        self.assertEqual(status, 401)
        self.assertEqual(body['received'], payload)

    def test_monitor_logs_metadata_and_preserves_body(self):
        payload = {'email': "admin'--", 'password': 'private-secret'}
        with self.assertLogs('aiwaf.payload', 'WARNING') as logs:
            status, body = self.send(payload, mode='monitor')
        self.assertEqual(status, 401)
        self.assertEqual(body['received'], payload)
        self.assertNotIn('private-secret', ''.join(logs.output))

    def test_off_and_exemption_forward_attack_body(self):
        for kwargs in [{'mode': 'off'}, {'exempt': True}]:
            status, body = self.send({'email': "admin'--"}, **kwargs)
            self.assertEqual(status, 401)
            self.assertEqual(body['received']['email'], "admin'--")

    def test_limits_block_and_monitor_without_truncation(self):
        payload = {'email': 'a' * 100}
        self.assertEqual(self.send(payload, max_bytes=16)[0], 413)
        with self.assertLogs('aiwaf.payload', 'WARNING'):
            status, body = self.send(payload, mode='monitor', max_bytes=16)
        self.assertEqual(status, 401)
        self.assertEqual(body['received'], payload)

    def test_query_and_urlencoded_form(self):
        self.assertEqual(self.send({}, query='q=union%20select%20password')[0], 403)
        self.assertEqual(self.send(b'email=admin%27--', content_type='application/x-www-form-urlencoded')[0], 403)

    def test_multipart_body_preserved(self):
        raw = b'--x\r\nContent-Disposition: form-data; name="file"\r\n\r\nadmin\x27--\r\n--x--\r\n'
        status, body = self.send(raw, content_type='multipart/form-data; boundary=x')
        self.assertEqual(status, 401)
        self.assertEqual(body['raw'], raw.decode())


def wire_body(payload):
    return payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def echo_body(raw, content_type):
    return {'received': json.loads(raw)} if content_type.startswith('application/json') else {'raw': raw.decode()}
