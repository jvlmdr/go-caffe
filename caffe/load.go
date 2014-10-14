package caffe

//	import (
//		"github.com/jvlmdr/go-cv/convfeat"
//		"github.com/jvlmdr/go-cv/feat"
//	)
//
//	func FromProto(net *NetParameter, output string, mean []float64) (feat.Image, error) {
//		if len(net.Input) != 1 {
//			return nil, fmt.Errorf("number of network inputs is not 1: %d", len(net.Input))
//		}
//
//		layer := layers[output]
//		if output == net.Input[0] {
//			// TODO: Scale by 255.0 and subtract mean!
//			return feat.NewRGB()
//		}
//	}
//
//	func fromProto() (feat.Real, error) {
//		outer :=
//
//		if len(layer.Bottom) != 1 {
//			return nil, fmt.Errorf("number of layer inputs is not 1: %d", len(layer.Bottom))
//		}
//		// Check if this layer has an in-place operation.
//	}
//
//	func layerToFunc(layer) (feat.Real, error) {
//		switch layer.GetType() {
//		case LayerParameter_CONVOLUTION:
//		case LayerParameter_LRN:
//		case LayerParameter_POOLING:
//		case LayerParameter_RELU:
//		}
//	}
//
//	func convLayerToFunc(layer) (feat.Real, error) {
//		if len(layer.Blobs) != 2 {
//			return nil, fmt.Errorf("number of convolution blobs is not 2: %d", len(layer.Blobs))
//		}
//		bank, err := filterBankFromBlob(layer.Blobs[0])
//		if err != nil {
//			return nil, err
//		}
//		bias, err := biasFromBlob(layer.Blobs[1])
//		if err != nil {
//			return nil, err
//		}
//		if bank.NumOut != len(bias) {
//			err := fmt.Errorf("different number of channels: filter output %d, bias %d", bank.NumOut, len(bias))
//			return nil, err
//		}
//		addBias := &AddConst
//		&feat.Compose{Outer: AddConst
//	}
//
//	func errIfWrongNumElems(blob *BlobProto) error {
//		var (
//			out = int(blob.GetNum())
//			width  = int(blob.GetWidth())
//			height = int(blob.GetHeight())
//			in  = int(blob.GetChannels())
//		)
//		if len(blob.Data) != out*width*height*in {
//			return fmt.Errorf(
//				"wrong number of elements for %dx%dx%dx%d: %d",
//				out, width, height, in, len(blob.Data),
//			)
//		}
//		return nil
//	}
//
//	func biasFromBlob(blob *BlobProto) ([]float64, error) {
//		var (
//			numOut = int(blob.GetNum())
//			width  = int(blob.GetWidth())
//			height = int(blob.GetHeight())
//			numIn  = int(blob.GetChannels())
//		)
//		if numOut != 1 || width != 1 || height != 1 {
//			err := fmt.Errorf("expect singleton dimensions: %dx%dx%dx%d", numOut, width, height, numIn)
//			return nil, err
//		}
//		if err := errIfWrongNumElems(blob); err != nil {
//			return nil, err
//		}
//		return toFloat64s(blob.Data), nil
//	}
//
//	func toFloat64s(x []float32) []float64 {
//		y := make([]float64, len(x))
//		for i, v := range x {
//			y[i] = float64(v)
//		}
//		return y
//	}
//
//	func filterBankFromBlob(blob *BlobProto) (*convfeat.FilterBankMulti, error) {
//		var (
//			numOut = int(blob.GetNum())
//			width  = int(blob.GetWidth())
//			height = int(blob.GetHeight())
//			numIn  = int(blob.GetChannels())
//		)
//		if err := errIfWrongNumElems(blob); err != nil {
//			return nil, err
//		}
//
//		filts := make([]*rimg64.Multi, numOut)
//		for i := range filts {
//			filts[i] = rimg64.NewMulti(width, height, numOut)
//		}
//		var ind int
//		for i := 0; i < numOut; i++ {
//			for p := 0; p < numIn; p++ {
//				for v := 0; v < height; v++ {
//					for u := 0; u < width; u++ {
//						//ind := ((i*numIn+p)*height+v)*width + u
//						// Flip filter.
//						filts[i].Set(width-1-u, height-1-v, p, float64(blob.Data[ind]))
//						ind++
//					}
//				}
//			}
//		}
//		field := image.Pt(width, height)
//		bank := &FilterBankMulti{field, numIn, filts}
//		return bank, nil
//	}
