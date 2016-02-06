package anagrammer

type Question struct {
	Q string   `json:"q"`
	A []string `json:"a"`
}
