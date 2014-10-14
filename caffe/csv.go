package caffe

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"github.com/jvlmdr/go-cv/rimg64"
)

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
