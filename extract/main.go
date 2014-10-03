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
	"regexp"
	"strconv"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s extract.py image model.txt weights mean.npy\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	if flag.NArg() != 5 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		scriptFile  = flag.Arg(0)
		imFile      = flag.Arg(1)
		modelFile   = flag.Arg(2)
		weightsFile = flag.Arg(3)
		meanFile    = flag.Arg(4)
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
	modify(model)
	fs, err := caffe.Extract(scriptFile, ims, "conv5", model, weightsFile, meanFile)
	if err != nil {
		log.Fatalln(err)
	}
	for i := range ims {
		fmt.Println(ims[i].Bounds().Size(), "->", fs[i].Size())
	}
}

func modify(model *caffe.NetParameter) {
	var subset []*caffe.LayerParameter
	for _, layer := range model.Layers {
		name := layer.GetName()
		if !keepLayer(name) {
			fmt.Println("excise layer:", name)
			continue
		}
		subset = append(subset, layer)
	}
	model.Layers = subset
}

func keepLayer(name string) bool {
	if name == "data" {
		return true
	}
	re := regexp.MustCompile(`\d+$`)
	numstr := re.FindString(name)
	if len(numstr) == 0 {
		return false // No number.
	}
	num, err := strconv.ParseInt(numstr, 10, 32)
	if err != nil {
		panic(fmt.Sprintf("not a number: %s", numstr))
	}
	if num > 5 {
		return false
	}
	return true
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
