package caffe

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"strconv"

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
		outputFiles[i] = path.Join(dir, fmt.Sprintf("feats-%03d.csv", i))
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
	for i := range feats {
		err := load(outputFiles[i], func(r io.ReadSeeker) (err error) {
			feats[i], err = readMultiCSV(r)
			return
		})
		if err != nil {
			return nil, err
		}
	}
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

func writeFileList(w io.Writer, inputs, outputs []string) error {
	if len(inputs) != len(outputs) {
		panic(fmt.Sprintf("different number of inputs and outputs: %d, %d", len(inputs), len(outputs)))
	}
	cw := csv.NewWriter(w)
	defer cw.Flush()
	for i := range inputs {
		if err := cw.Write([]string{inputs[i], outputs[i]}); err != nil {
			return err
		}
	}
	return nil
}

func readMultiCSV(r io.ReadSeeker) (*rimg64.Multi, error) {
	m, n, c, err := readMultiDimsCSV(r)
	if err != nil {
		return nil, err
	}
	if _, err := r.Seek(0, 0); err != nil {
		return nil, err
	}
	f := rimg64.NewMulti(m, n, c)
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = 4
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		i, err := strconv.ParseInt(rec[0], 10, 32)
		if err != nil {
			return nil, err
		}
		j, err := strconv.ParseInt(rec[1], 10, 32)
		if err != nil {
			return nil, err
		}
		k, err := strconv.ParseInt(rec[2], 10, 32)
		if err != nil {
			return nil, err
		}
		x, err := strconv.ParseFloat(rec[3], 64)
		if err != nil {
			return nil, err
		}
		f.Set(int(i), int(j), int(k), x)
	}
	return f, nil
}

func readMultiDimsCSV(r io.Reader) (int, int, int, error) {
	var m, n, c int
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = 4
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, 0, 0, err
		}
		i, err := strconv.ParseInt(rec[0], 10, 32)
		if err != nil {
			return 0, 0, 0, err
		}
		j, err := strconv.ParseInt(rec[1], 10, 32)
		if err != nil {
			return 0, 0, 0, err
		}
		k, err := strconv.ParseInt(rec[2], 10, 32)
		if err != nil {
			return 0, 0, 0, err
		}
		m = max(m, int(i)+1)
		n = max(n, int(j)+1)
		c = max(c, int(k)+1)
	}
	return m, n, c, nil
}

func max(a, b int) int {
	if b > a {
		return b
	}
	return a
}
