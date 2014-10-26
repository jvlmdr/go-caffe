package caffe

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/convfeat"
	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-cv/rimg64"
)

func FromProto(net *NetParameter, output string, mean []float64) (featset.Image, error) {
	if len(net.Input) != 1 {
		return nil, fmt.Errorf("number of network inputs is not 1: %d", len(net.Input))
	}
	phi, _, err := fromProto(net, output)
	if err != nil {
		return nil, err
	}
	// Scale by 255 and subtract mean.
	scale := new(convfeat.Scale)
	*scale = 255
	subMean := new(convfeat.AddConst)
	*subMean = neg(mean)
	reorder := &featset.SelectChannels{Channels: []int{2, 1, 0}}
	preproc := &featset.ComposeImage{
		Outer: &featset.Compose{
			Outer: reorder,
			Inner: &featset.Compose{Outer: subMean, Inner: scale}},
		Inner: new(featset.RGB),
	}
	return &featset.ComposeImage{phi, preproc}, nil
}

func fromProto(net *NetParameter, name string) (featset.Real, int, error) {
	if name == net.Input[0] {
		// We've struck pixels!
		return nil, 3, nil
	}
	layer := layerByName(net, name)
	if err := errIfNotOneInput(layer); err != nil {
		return nil, 0, err
	}
	// Drill down.
	below, in, err := fromProto(net, layer.Bottom[0])
	if err != nil {
		return nil, 0, err
	}

	// Convert this layer into a real-valued transform.
	var (
		phi featset.Real
		out int
	)
	phi, out, err = layerToFunc(layer, in)
	if err != nil {
		return nil, 0, fmt.Errorf("layer %s: %v", name, err)
	}
	// Check if this layer has an in-place operation.
	loop, err := findLoop(net, name)
	if err != nil {
		return nil, 0, err
	}
	if loop != "" {
		// Inputs to loop are outputs of layer.
		// Number of outputs should map.
		outer, loopOut, err := layerToFunc(layerByName(net, loop), out)
		if err != nil {
			return nil, 0, fmt.Errorf("layer %s: %v", loop, err)
		}
		if loopOut != out {
			return nil, 0, fmt.Errorf("in-place layer changes dimension: from %d to %d", out, loopOut)
		}
		phi = &featset.Compose{Outer: outer, Inner: phi}
	}
	if below != nil {
		phi = &featset.Compose{Outer: phi, Inner: below}
	}
	return phi, out, nil
}

func errIfNotOneInput(layer *LayerParameter) error {
	if len(layer.Bottom) != 1 {
		return fmt.Errorf("number of layer inputs is not 1: %d", len(layer.Bottom))
	}
	return nil
}

func errIfDimsNotEq(want, got BlobDims) error {
	if want != got {
		return fmt.Errorf("blob dims: expect %v, found %v", want, got)
	}
	return nil
}

func findLoop(net *NetParameter, name string) (string, error) {
	var subset []string
	for _, l := range net.Layers {
		if len(l.Top) != 1 {
			continue
		}
		if len(l.Bottom) != 1 {
			continue
		}
		if l.Top[0] == name && l.Bottom[0] == name {
			subset = append(subset, l.GetName())
		}
	}
	if len(subset) == 0 {
		return "", nil
	}
	if len(subset) > 1 {
		return "", fmt.Errorf("multiple loops for %s: %v", name, subset)
	}
	return subset[0], nil
}

func neg(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, xi := range x {
		y[i] = -xi
	}
	return y
}

// Converts a single layer into a real feature transform.
func layerToFunc(layer *LayerParameter, in int) (featset.Real, int, error) {
	switch t := layer.GetType(); t {
	case LayerParameter_CONVOLUTION:
		return convLayerToFunc(layer, in)
	case LayerParameter_LRN:
		return lrnLayerToFunc(layer, in)
	case LayerParameter_POOLING:
		return poolLayerToFunc(layer, in)
	case LayerParameter_RELU:
		return reluLayerToFunc(layer, in)
	default:
		return nil, 0, fmt.Errorf("unknown layer type: %s", t.String())
	}
}

func convLayerToFunc(layer *LayerParameter, in int) (featset.Real, int, error) {
	param := layer.GetConvolutionParam()
	if param.GetPad() != 0 {
		return nil, 0, fmt.Errorf("non-zero pad: %d", param.GetPad())
	}
	var (
		out    = int(param.GetNumOutput())
		size   = int(param.GetKernelSize())
		stride = int(param.GetStride())
		groups = int(param.GetGroup())
	)
	if len(layer.Blobs) != 2 {
		return nil, 0, fmt.Errorf("number of convolution blobs is not 2: %d", len(layer.Blobs))
	}
	var conv featset.Real
	if groups <= 1 {
		dims := BlobDims{Width: size, Height: size, In: in, Out: out}
		bank, err := filterBankFromBlob(layer.Blobs[0], dims)
		if err != nil {
			return nil, 0, err
		}
		conv = &convfeat.ConvMulti{Stride: stride, Filters: bank}
	} else {
		dims := BlobDims{Width: size, Height: size, In: in / groups, Out: out}
		banks, err := filterBanksFromBlob(layer.Blobs[0], dims, groups)
		if err != nil {
			return nil, 0, err
		}
		phis := make([]featset.Real, groups)
		for i := range banks {
			phis[i] = &featset.Compose{
				Outer: &convfeat.ConvMulti{Stride: stride, Filters: banks[i]},
				Inner: &featset.ChannelInterval{i * dims.In, (i + 1) * dims.In},
			}
		}
		conv = &featset.Concat{featset.RealSlice(phis)}
	}
	bias, err := biasFromBlob(layer.Blobs[1], out)
	if err != nil {
		return nil, 0, err
	}
	if len(bias) != out {
		err := fmt.Errorf("number of channels for bias: expect %d, found %d", out, len(bias))
		return nil, 0, err
	}
	addBias := new(convfeat.AddConst)
	*addBias = convfeat.AddConst(bias)
	return &featset.Compose{Outer: addBias, Inner: conv}, out, nil
}

func errIfWrongNumElems(dims BlobDims, blob *BlobProto) error {
	if len(blob.Data) != dims.NumElems() {
		return fmt.Errorf("wrong number of elements for %v: %d", dims, len(blob.Data))
	}
	return nil
}

func biasFromBlob(blob *BlobProto, n int) ([]float64, error) {
	dims := BlobDims{Width: n, Height: 1, In: 1, Out: 1}
	got := blobDims(blob)
	if err := errIfDimsNotEq(dims, got); err != nil {
		return nil, err
	}
	if err := errIfWrongNumElems(dims, blob); err != nil {
		return nil, err
	}
	return toFloat64s(blob.Data), nil
}

func toFloat64s(x []float32) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = float64(v)
	}
	return y
}

type BlobDims struct{ Width, Height, In, Out int }

func (d BlobDims) NumElems() int {
	return d.Width * d.Height * d.In * d.Out
}

func blobDims(blob *BlobProto) BlobDims {
	return BlobDims{
		Width:  int(blob.GetWidth()),
		Height: int(blob.GetHeight()),
		In:     int(blob.GetChannels()),
		Out:    int(blob.GetNum()),
	}
}

// dims is the expected dimensions of the blob.
// If the layer as a whole maps m channels to n channels,
// then the blob will be have m/groups inputs and n outputs.
// The first n/groups filters are applied to the first m/groups channels and so on.
func filterBanksFromBlob(blob *BlobProto, dims BlobDims, groups int) ([]*convfeat.FilterBankMulti, error) {
	if err := errIfDimsNotEq(dims, blobDims(blob)); err != nil {
		return nil, err
	}
	if err := errIfWrongNumElems(dims, blob); err != nil {
		return nil, err
	}

	outPerGroup := dims.Out / groups
	banks := make([]*convfeat.FilterBankMulti, groups)
	var ind int
	for j := range banks {
		filts := make([]*rimg64.Multi, outPerGroup)
		for i := range filts {
			filts[i] = rimg64.NewMulti(dims.Width, dims.Height, dims.In)
		}
		for i := 0; i < outPerGroup; i++ {
			for p := 0; p < dims.In; p++ {
				for v := 0; v < dims.Height; v++ {
					for u := 0; u < dims.Width; u++ {
						//ind := ((i*dims.In+p)*dims.Height+v)*dims.Width + u
						//	// Flip filter.
						//	filts[i].Set(dims.Width-1-u, dims.Height-1-v, p, float64(blob.Data[ind]))
						filts[i].Set(u, v, p, float64(blob.Data[ind]))
						ind++
					}
				}
			}
		}
		field := image.Pt(dims.Width, dims.Height)
		banks[j] = &convfeat.FilterBankMulti{field, dims.In, filts}
	}
	return banks, nil
}

func filterBankFromBlob(blob *BlobProto, dims BlobDims) (*convfeat.FilterBankMulti, error) {
	if err := errIfDimsNotEq(dims, blobDims(blob)); err != nil {
		return nil, err
	}
	if err := errIfWrongNumElems(dims, blob); err != nil {
		return nil, err
	}

	filts := make([]*rimg64.Multi, dims.Out)
	for i := range filts {
		filts[i] = rimg64.NewMulti(dims.Width, dims.Height, dims.In)
	}
	var ind int
	for i := 0; i < dims.Out; i++ {
		for p := 0; p < dims.In; p++ {
			for v := 0; v < dims.Height; v++ {
				for u := 0; u < dims.Width; u++ {
					//ind := ((i*dims.In+p)*dims.Height+v)*dims.Width + u
					//	// Flip filter.
					//	filts[i].Set(dims.Width-1-u, dims.Height-1-v, p, float64(blob.Data[ind]))
					filts[i].Set(u, v, p, float64(blob.Data[ind]))
					ind++
				}
			}
		}
	}
	field := image.Pt(dims.Width, dims.Height)
	bank := &convfeat.FilterBankMulti{field, dims.In, filts}
	return bank, nil
}

func lrnLayerToFunc(layer *LayerParameter, in int) (featset.Real, int, error) {
	param := layer.GetLrnParam()
	if param.GetNormRegion() != LRNParameter_ACROSS_CHANNELS {
		return nil, 0, fmt.Errorf("normalization region: %s", param.GetNormRegion().String())
	}
	phi := &convfeat.AdjChanNorm{
		Num:   int(param.GetLocalSize()),
		K:     1,
		Alpha: float64(param.GetAlpha()) / float64(param.GetLocalSize()),
		Beta:  float64(param.GetBeta()),
	}
	return phi, in, nil
}

func poolLayerToFunc(layer *LayerParameter, in int) (featset.Real, int, error) {
	param := layer.GetPoolingParam()
	if param.GetPool() != PoolingParameter_MAX {
		return nil, 0, fmt.Errorf("pool type: %s", param.GetPool().String())
	}
	if param.GetPad() != 0 {
		return nil, 0, fmt.Errorf("non-zero pad: %d", param.GetPad())
	}
	side := int(param.GetKernelSize())
	phi := &convfeat.MaxPool{
		Field:  image.Pt(side, side),
		Stride: int(param.GetStride()),
	}
	return phi, in, nil
}

func reluLayerToFunc(layer *LayerParameter, in int) (featset.Real, int, error) {
	param := layer.GetReluParam()
	if param.GetNegativeSlope() != 0 {
		return nil, 0, fmt.Errorf("non-zero negative slope: %g", param.GetNegativeSlope())
	}
	return new(convfeat.PosPart), in, nil
}
