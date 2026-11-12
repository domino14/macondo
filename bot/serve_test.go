package bot

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
)

// These need a NATS server on the default URL; CI has none, so they skip.
func natsConn(t *testing.T) *nats.Conn {
	nc, err := nats.Connect(nats.DefaultURL, nats.Timeout(time.Second))
	if err != nil {
		t.Skip("no local NATS server:", err)
	}
	t.Cleanup(nc.Close)
	return nc
}

func subject(t *testing.T) string {
	return fmt.Sprintf("bottest.%s.%d", t.Name(), time.Now().UnixNano())
}

// collect records the subjects published on prefix.>.
func collect(t *testing.T, nc *nats.Conn, prefix string) func() []string {
	ch := make(chan string, 64)
	sub, err := nc.Subscribe(prefix+".>", func(m *nats.Msg) { ch <- m.Subject })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sub.Unsubscribe() })
	if err := nc.Flush(); err != nil {
		t.Fatal(err)
	}
	return func() []string {
		var got []string
		timeout := time.After(time.Second)
		for {
			select {
			case s := <-ch:
				got = append(got, s)
			case <-timeout:
				return got
			}
		}
	}
}

// startServe runs serve on nc and returns a function that
// stops it and waits for it to return.
func startServe(t *testing.T, nc *nats.Conn, channel string, handle nats.MsgHandler, timeout time.Duration) func() time.Duration {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serve(ctx, nc, channel, handle, timeout) }()
	// Wait until it is subscribed.
	deadline := time.Now().Add(2 * time.Second)
	for nc.NumSubscriptions() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	return func() time.Duration {
		start := time.Now()
		cancel()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		return time.Since(start)
	}
}

// Requests already delivered when the bot is told to stop are still
// answered, and their moves sent, before it returns.
func TestServeAnswersDeliveredRequests(t *testing.T) {
	channel := subject(t)
	out := natsConn(t)
	got := collect(t, out, channel+".move")

	var started atomic.Int32
	var handled atomic.Int32
	botConn := natsConn(t)
	stop := startServe(t, botConn, channel, func(m *nats.Msg) {
		started.Add(1)
		time.Sleep(200 * time.Millisecond) // computing a move
		handled.Add(1)
		// Sent on the bot's own connection, as the real handler does.
		botConn.Publish(channel+".move."+string(m.Data), nil)
	}, 5*time.Second)

	for i := range 3 {
		if err := out.Publish(channel, []byte(fmt.Sprint(i))); err != nil {
			t.Fatal(err)
		}
	}
	out.Flush()
	for started.Load() == 0 {
		time.Sleep(5 * time.Millisecond)
	}
	stop() // while the first is being computed

	if handled.Load() != 3 {
		t.Fatalf("handled %d of 3 requests", handled.Load())
	}
	if moves := got(); len(moves) != 3 {
		t.Fatalf("got %d of 3 moves: %v", len(moves), moves)
	}
}

// Once the bot is draining, new requests go to the other bots in the queue
// group.
func TestServeDrainingSendsNewRequestsElsewhere(t *testing.T) {
	channel := subject(t)
	out := natsConn(t)

	release := make(chan struct{})
	var mine atomic.Int32
	stop := startServe(t, natsConn(t), channel, func(m *nats.Msg) {
		mine.Add(1)
		<-release
	}, 5*time.Second)

	var other atomic.Int32
	otherSub, err := out.QueueSubscribe(channel, "bot_queue", func(m *nats.Msg) { other.Add(1) })
	if err != nil {
		t.Fatal(err)
	}
	defer otherSub.Unsubscribe()
	out.Flush()

	// Keep this bot busy, then stop it.
	for mine.Load() == 0 {
		out.Publish(channel, []byte("x"))
		out.Flush()
		time.Sleep(10 * time.Millisecond)
	}
	stopped := make(chan struct{})
	go func() { stop(); close(stopped) }()
	time.Sleep(100 * time.Millisecond) // the drain has started

	before := mine.Load()
	for range 10 {
		out.Publish(channel, []byte("y"))
	}
	out.Flush()
	time.Sleep(200 * time.Millisecond)
	close(release)
	<-stopped
	if mine.Load() != before {
		t.Fatalf("draining bot took %d new requests", mine.Load()-before)
	}
	if other.Load() < 10 {
		t.Fatalf("the other bot got %d of 10 new requests", other.Load())
	}
}

// A request that never finishes doesn't hold the shutdown past the timeout.
func TestServeGivesUpAtTimeout(t *testing.T) {
	channel := subject(t)
	out := natsConn(t)

	release := make(chan struct{})
	defer close(release)
	var started atomic.Int32
	stop := startServe(t, natsConn(t), channel, func(m *nats.Msg) {
		started.Add(1)
		<-release
	}, 300*time.Millisecond)

	out.Publish(channel, []byte("x"))
	out.Flush()
	for started.Load() == 0 {
		time.Sleep(5 * time.Millisecond)
	}
	if took := stop(); took > 2*time.Second {
		t.Fatalf("stopping took %v", took)
	}
}
