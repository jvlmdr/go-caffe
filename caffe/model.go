package caffe

import (
	"fmt"
	"image"
)

func LayerRate(net *NetParameter, name string) int {
	return rateHelper(net, name, 1)
}

func rateHelper(net *NetParameter, name string, prod int) int {
	if name == "" {
		panic("no layer name given")
	}
	if isInput(net, name) {
		return prod
	}
	layer := layerByName(net, name)
	if layer == nil {
		panic(fmt.Sprintf("could not find layer: %s", name))
	}
	var stride int
	switch *layer.Type {
	case LayerParameter_CONVOLUTION:
		stride = int(layer.GetConvolutionParam().GetStride())
	case LayerParameter_POOLING:
		stride = int(layer.GetPoolingParam().GetStride())
	case LayerParameter_LRN:
		stride = 1
	default:
		typename := LayerParameter_LayerType_name[int32(*layer.Type)]
		panic(fmt.Sprintf("do not handle layer type: %s", typename))
	}
	bottoms := layer.GetBottom()
	if len(bottoms) != 1 {
		panic(fmt.Sprintf("layer %s does not have one input: %v", name, bottoms))
	}
	return rateHelper(net, bottoms[0], stride*prod)
}

func isInput(net *NetParameter, name string) bool {
	for _, input := range net.Input {
		if name == input {
			return true
		}
	}
	return false
}

func layerByName(net *NetParameter, name string) *LayerParameter {
	for _, layer := range net.Layers {
		if layer.GetName() == name {
			return layer
		}
	}
	return nil
}

func LayerField(net *NetParameter, name string) image.Point {
	_, p := fieldHelper(net, name)
	return p
}

func fieldHelper(net *NetParameter, name string) (rate int, field image.Point) {
	if name == "" {
		panic("no layer name given")
	}
	if isInput(net, name) {
		return 1, image.Pt(1, 1)
	}
	layer := layerByName(net, name)
	if layer == nil {
		panic(fmt.Sprintf("could not find layer: %s", name))
	}
	var (
		k int
		p image.Point
	)
	switch *layer.Type {
	case LayerParameter_CONVOLUTION:
		k = int(layer.GetConvolutionParam().GetStride())
		p = layer.GetConvolutionParam().Kernel()
	case LayerParameter_POOLING:
		k = int(layer.GetPoolingParam().GetStride())
		p = layer.GetPoolingParam().Kernel()
	case LayerParameter_LRN:
		k, p = 1, image.Pt(1, 1)
	default:
		typename := LayerParameter_LayerType_name[int32(*layer.Type)]
		panic(fmt.Sprintf("do not handle layer type: %s", typename))
	}
	bottoms := layer.GetBottom()
	if len(bottoms) != 1 {
		panic(fmt.Sprintf("layer %s does not have one input: %v", name, bottoms))
	}
	s, n := fieldHelper(net, bottoms[0])
	return k * s, p.Sub(image.Pt(1, 1)).Mul(s).Add(n)
}

func (param ConvolutionParameter) Kernel() image.Point {
	if param.KernelSize != nil {
		p := int(*param.KernelSize)
		return image.Pt(p, p)
	}
	px := int(param.GetKernelW())
	py := int(param.GetKernelH())
	return image.Pt(px, py)
}

func (param PoolingParameter) Kernel() image.Point {
	if param.KernelSize != nil {
		p := int(*param.KernelSize)
		return image.Pt(p, p)
	}
	px := int(param.GetKernelW())
	py := int(param.GetKernelH())
	return image.Pt(px, py)
}
