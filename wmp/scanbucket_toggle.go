package wmp

// useScanBucketAsm chooses between the pure-Go scanBucket (inlined
// at every call site by the Go compiler) and the hand-written ABI0
// assembly stub scanBucketAsm (in scanbucket_arm64.s /
// scanbucket_amd64.s).
//
// Default: false. The asm path measured ~14% SLOWER than the
// inlined Go path on Apple M2 (BenchmarkGetWordEntry: 7.3 ns vs
// 6.4 ns). The reason is the same as for the BitRack mix asm
// experiment: scanBucket is small enough that the Go inliner
// pulls it into each caller (cost 72, under the 80 budget) — so
// the inlined version sits directly inside the surrounding loop
// with no call overhead, while the asm version must go through an
// ABI0 → ABIInternal bridge that shuffles registers in and out of
// the stack frame on every call.
//
// The toggle and the .s files are kept in-tree as a worked example
// of when assembly stubs DON'T help in Go. To revisit (e.g., if a
// future Go release stops inlining scanBucket, or if we add an
// ABIInternal asm variant), flip this constant to true.
const useScanBucketAsm = false
