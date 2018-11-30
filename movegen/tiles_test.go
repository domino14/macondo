package movegen

// import "testing"

// func TestInitialize(t *testing.T) {
// 	b := &Bag{}
// 	b.Init()
// 	if len(b.bag) != 100 {
// 		t.Errorf("Length is %v, expected 100", len(b.bag))
// 	}
// }

// func TestDraw(t *testing.T) {
// 	b := &Bag{}
// 	b.Init()
// 	letters := b.Draw(7)
// 	if len(letters) != 7 {
// 		t.Errorf("Length was %v, expected 7", len(letters))
// 	}
// 	if len(b.bag) != 93 {
// 		t.Errorf("Length was %v, expected 93", len(b.bag))
// 	}
// }

// func TestExchange(t *testing.T) {
// 	b := &Bag{}
// 	b.Init()
// 	letters := b.Draw(7)
// 	newLetters := b.Exchange(letters[:5])
// 	if len(newLetters) != 5 {
// 		t.Errorf("Length was %v, expected 5", len(newLetters))
// 	}
// 	if len(b.bag) != 93 {
// 		t.Errorf("Length was %v, expected 93", len(b.bag))
// 	}
// }
