package caffe

func SubsetForOutput(src *NetParameter, output string) *NetParameter {
	subset := layersBefore(src, output)
	var layers []*LayerParameter
	for _, layer := range src.Layers {
		if subset[layer.GetName()] {
			layers = append(layers, layer)
		}
	}
	dst := new(NetParameter)
	*dst = *src
	dst.Layers = layers
	return dst
}

func layersBefore(net *NetParameter, output string) map[string]bool {
	subset := make(map[string]bool)
	setLayersBefore(net, output, subset)
	return subset
}

func layersByTop(net *NetParameter, name string) []string {
	var layers []string
	for _, l := range net.Layers {
		for _, t := range l.Top {
			if t == name {
				layers = append(layers, l.GetName())
			}
		}
	}
	return layers
}

func setLayersBefore(net *NetParameter, name string, subset map[string]bool) {
	if isInput(net, name) {
		return
	}
	if subset[name] {
		// This layer has already been visited.
		return
	}
	subset[name] = true
	layer := layerByName(net, name)
	if layer == nil {
		panic("could not find layer: " + name)
	}
	// Find all layers from which this layer takes its input.
	for _, child := range layer.Bottom {
		setLayersBefore(net, child, subset)
	}
	// Find all layers which have this layer as their output.
	for _, child := range layersByTop(net, name) {
		setLayersBefore(net, child, subset)
	}
}
