const { extractFeatures, init } = require('./lib/featureUtils');
const anomalyDetector = require('./lib/anomalyDetector');

// Initialize components
init();
anomalyDetector.init();

// Test cases representing different types of requests
const testCases = [
  {
    name: "Normal request",
    req: {
      path: "/api/users",
      ip: "192.168.1.200",
      res: { statusCode: 200 },
      headers: { 'x-response-time': '0.123' }
    }
  },
  {
    name: "Suspicious PHP request",
    req: {
      path: "/admin.php",
      ip: "192.168.1.201",
      res: { statusCode: 404 },
      headers: { 'x-response-time': '0.015' }
    }
  },
  {
    name: "Multiple malicious keywords",
    req: {
      path: "/wp-admin/.env.bak",
      ip: "192.168.1.202",
      res: { statusCode: 404 },
      headers: { 'x-response-time': '0.008' }
    }
  },
  {
    name: "Very long path",
    req: {
      path: "/this-is-a-very-long-suspicious-path-that-might-be-an-attack-vector-1234567890",
      ip: "192.168.1.203",
      res: { statusCode: 404 },
      headers: { 'x-response-time': '2.5' }
    }
  }
];

async function runTests() {
  console.log('🔍 Testing Anomaly Detection\n');
  console.log('Model Info:', anomalyDetector.getModelInfo());
  console.log();

  for (const testCase of testCases) {
    try {
      // Simulate multiple requests for burst calculation
      for (let i = 0; i < 3; i++) {
        const features = await extractFeatures(testCase.req);
        const isAnomalous = anomalyDetector.isAnomalous(features);
        
        if (i === 2) { // Only show result for the last request
          console.log(`${testCase.name}:`);
          console.log(`  Path: ${testCase.req.path}`);
          console.log(`  Features: [${features.join(', ')}]`);
          console.log(`  Anomalous: ${isAnomalous ? '🚨 YES' : '✅ NO'}`);
          console.log();
        }
        
        // Small delay to simulate real timing
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } catch (err) {
      console.error(`Error testing ${testCase.name}:`, err.message);
    }
  }
}

runTests().catch(console.error);
