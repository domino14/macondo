package main

import "fmt"
import "unicode/utf8"

func main() {
     ano := "AÑO";
     for i := 0; i < len(ano); i++ {
     	 fmt.Println(ano[i])
}
     fmt.Println(len(ano))
     fmt.Println(utf8.RuneCountInString(ano))
     runearr := []rune(ano)
     
     // 61 c3    
     for i := 0; i < len(runearr); i++ {
     	 fmt.Printf("%c %v\n", runearr[i], runearr[i])
	}
}