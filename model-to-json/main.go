package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
	"github.com/jvlmdr/go-file/fileutil"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s in.txt out.json\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		inFile  = flag.Arg(0)
		outFile = flag.Arg(1)
	)

	modelStr, err := ioutil.ReadFile(inFile)
	if err != nil {
		log.Fatalln(err)
	}
	model := new(caffe.NetParameter)
	if err := proto.UnmarshalText(string(modelStr), model); err != nil {
		log.Fatalln(err)
	}
	if err := fileutil.SaveJSON(outFile, model); err != nil {
		log.Fatalln(err)
	}
}
