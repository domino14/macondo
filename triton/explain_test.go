package triton

import (
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestExplainModelNotServed(t *testing.T) {
	c := &TritonClient{serverURL: "localhost:8001", modelName: "macondo-tf", modelVersion: "1"}
	notReady := status.Error(codes.Unavailable,
		"Request for unknown model: 'macondo-tf' version 1 is not at ready state")
	if got := c.explain(notReady).Error(); !strings.Contains(got, "triton-models.sh load macondo-tf") ||
		!strings.Contains(got, "not at ready state") {
		t.Errorf("no hint: %s", got)
	}
	other := status.Error(codes.Unavailable, "connection refused")
	if got := c.explain(other); got != other {
		t.Errorf("hint added to an unrelated error: %v", got)
	}
}
