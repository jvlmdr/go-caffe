package caffe

import (
	"fmt"
	"image"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-cv/rimg64"
)

func Extract(scriptFile string, ims []image.Image, layer string, model *NetParameter, weightsFile, meanFile string) ([]*rimg64.Multi, error) {
	dir, err := ioutil.TempDir("", "tmp-")
	if err != nil {
		return nil, err
	}
	defer remove(dir)
	// Save images to files.
	var (
		inputFiles  = make([]string, len(ims))
		outputFiles = make([]string, len(ims))
	)
	for i, im := range ims {
		inputFiles[i] = path.Join(dir, fmt.Sprintf("image-%03d.png", i))
		outputFiles[i] = path.Join(dir, fmt.Sprintf("feats-%03d.multi", i))
		err := save(inputFiles[i], func(w io.Writer) error { return png.Encode(w, im) })
		if err != nil {
			return nil, err
		}
	}
	// Save list of files to a file.
	listFile := path.Join(dir, "files.csv")
	err = save(listFile, func(w io.Writer) error { return writeFileList(w, inputFiles, outputFiles) })
	if err != nil {
		return nil, err
	}
	// Save parameters to file.
	modelFile := path.Join(dir, "model.txt")
	err = save(modelFile, func(w io.Writer) error { return proto.MarshalText(w, model) })
	if err != nil {
		return nil, err
	}

	// Invoke Python program.
	cmd := exec.Command("python", scriptFile, modelFile, weightsFile, meanFile, layer, listFile)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Read output from CSV files.
	feats := make([]*rimg64.Multi, len(ims))
	log.Println("load images")
	for i := range feats {
		err := load(outputFiles[i], func(r io.ReadSeeker) (err error) {
			data, err := ioutil.ReadAll(r)
			if err != nil {
				return err
			}
			msg := new(Multi)
			if err := proto.Unmarshal(data, msg); err != nil {
				return err
			}
			feats[i] = multiFromProto(msg)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	log.Println("done: load images")
	return feats, nil
}

func remove(dir string) {
	err := os.RemoveAll(dir)
	if err != nil {
		log.Println("remove temp dir:", err)
	}
}

func save(fname string, write func(w io.Writer) error) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return write(file)
}

func load(fname string, read func(r io.ReadSeeker) error) error {
	file, err := os.Open(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return read(file)
}

func max(a, b int) int {
	if b > a {
		return b
	}
	return a
}
