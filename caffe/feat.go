package caffe

import (
	"image"
	"path"

	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
)

func init() {
	feat.RegisterImage("caffe", func() feat.Image { return new(Feature) })
}

var (
	ExtractScript string
	ModelsDir     string
	MeanFile      string
)

// Feature describes a pre-trained Caffe feature.
type Feature struct {
	// The pre-trained weights are expected at
	//  [ModelsDir]/[WeightsName]/[WeightsName].caffemodel
	WeightsName string
	// The model as protocol buffer text.
	Model *NetParameter
	// Name of layer to take as output.
	Layer string
}

func (phi *Feature) Rate() int {
	return LayerRate(phi.Model, phi.Layer)
}

func (phi *Feature) Apply(im image.Image) (*rimg64.Multi, error) {
	feats, err := phi.Map([]image.Image{im})
	if err != nil {
		return nil, err
	}
	return feats[0], nil
}

func (phi *Feature) Map(ims []image.Image) ([]*rimg64.Multi, error) {
	return Extract(ExtractScript, ims, phi.Layer, phi.Model, weightsFile(phi.WeightsName), MeanFile)
}

// Returns "[ModelsDir]/[name]/[name].caffemodel".
func weightsFile(name string) string {
	return path.Join(ModelsDir, name, name+".caffemodel")
}
