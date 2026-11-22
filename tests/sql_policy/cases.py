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
        self.assertEqual(self.send({}, query='q=1%20union%20select%20password')[0], 403)
        self.assertEqual(self.send(b'email=admin%27--', content_type='application/x-www-form-urlencoded')[0], 403)

    def test_multipart_body_preserved(self):
        raw = b'--x\r\nContent-Disposition: form-data; name="file"\r\n\r\no\x27reilly\r\n--x--\r\n'
        status, body = self.send(raw, content_type='multipart/form-data; boundary=x')
        self.assertEqual(status, 401)
        self.assertEqual(body['raw'], raw.decode())

    def test_multipart_parts_are_checked(self):
        multipart = 'multipart/form-data; boundary=x'
        part = lambda disposition, content: (
            b'--x\r\nContent-Disposition: form-data; ' + disposition + b'\r\n\r\n' + content + b'\r\n--x--\r\n')
        for raw, rule in [
            (part(b'name="email"', b"admin\x27--"), 'sql_'),
            (part(b'name="f"; filename="shell.php"', b'GIF89a'), 'rce_script_upload'),
            (part(b'name="f"; filename="a.jpg"', b'\xff\xd8\xff<?php system($_GET["c"]); ?>'), 'rce_php'),
            (part(b'name="__proto__[admin]"', b'1'), 'rce_prototype_pollution'),
        ]:
            with self.subTest(rule=rule):
                status, body = self.send(raw, content_type=multipart)
                self.assertEqual(status, 403)
                self.assertTrue(body['error'].startswith(rule), body)
        # A large upload is checked by its start and reaches the handler whole, not refused.
        upload = part(b'name="f"; filename="big.txt"', b'ordinary text ' * 20000)
        status, body = self.send(upload, content_type=multipart, max_bytes=4096)
        self.assertEqual(status, 401)
        self.assertEqual(len(body['raw']), len(upload))

    def test_bodies_in_an_encoding_that_cant_be_read_are_refused(self):
        self.assertEqual(self.send({'email': 'x'}, encoding='br')[0], 413)
        self.assertEqual(self.send({'email': 'x'}, encoding='identity')[0], 401)


def wire_body(payload):
    return payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def echo_body(raw, content_type):
    return {'received': json.loads(raw)} if content_type.startswith('application/json') else {'raw': raw.decode()}
