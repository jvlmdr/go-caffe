package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"os"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s extract.py image model.txt layer weights mean.npy\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	if flag.NArg() != 6 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		scriptFile  = flag.Arg(0)
		imFile      = flag.Arg(1)
		modelFile   = flag.Arg(2)
		output      = flag.Arg(3)
		weightsFile = flag.Arg(4)
		meanFile    = flag.Arg(5)
	)

	im, err := readImage(imFile)
	if err != nil {
		log.Fatalln(err)
	}
	ims := []image.Image{im}

	modelStr, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatalln(err)
	}
	model := new(caffe.NetParameter)
	if err := proto.UnmarshalText(string(modelStr), model); err != nil {
		log.Fatalln(err)
	}
	model = caffe.SubsetForOutput(model, output)
	fs, err := caffe.Extract(scriptFile, ims, output, model, weightsFile, meanFile)
	if err != nil {
		log.Fatalln(err)
	}
	f := fs[0]
	log.Println(im.Bounds().Size(), "->", f.Size())
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				fmt.Printf("%d,%d,%d,%g\n", i, j, k, f.At(i, j, k))
			}
		}
	}
}

func readImage(fname string) (image.Image, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return im, nil
}
