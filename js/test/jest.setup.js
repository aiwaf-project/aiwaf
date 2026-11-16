// The regular middleware's tests run the regular chain whatever aiwaf-wasm is installed.
// Tests of the automatic native default unset this themselves.
if (process.env.AIWAF_NATIVE_PIPELINE === undefined) process.env.AIWAF_NATIVE_PIPELINE = 'false';
