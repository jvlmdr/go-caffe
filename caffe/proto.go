package caffe

import (
	//	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-cv/rimg64"
)

//	func imageToProto(f *rimg64.Image) *Image {
//		n := f.Width * f.Height
//		pixels := make([]*Image_Pixel, 0, n)
//		for i := 0; i < f.Width; i++ {
//			for j := 0; j < f.Height; j++ {
//				pixel := &Image_Pixel{
//					X:     proto.Int(i),
//					Y:     proto.Int(j),
//					Value: proto.Float64(f.At(i, j)),
//				}
//				pixels = append(pixels, pixel)
//			}
//		}
//		return &Image{
//			Width:  proto.Int(f.Width),
//			Height: proto.Int(f.Height),
//			Pixels: pixels,
//		}
//	}
//
//	func imageFromProto(proto *Image) *rimg64.Image {
//		var (
//			width  = int(proto.GetWidth())
//			height = int(proto.GetHeight())
//		)
//		f := rimg64.New(width, height)
//		for _, pixel := range proto.Pixels {
//			var (
//				i = int(pixel.GetX())
//				j = int(pixel.GetY())
//			)
//			f.Set(i, j, pixel.GetValue())
//		}
//		return f
//	}

//	func multiToProto(f *rimg64.Multi) *Multi {
//		n := f.Width * f.Height * f.Channels
//		pixels := make([]*Multi_Pixel, 0, n)
//		for i := 0; i < f.Width; i++ {
//			for j := 0; j < f.Height; j++ {
//				for k := 0; k < f.Channels; k++ {
//					pixel := &Multi_Pixel{
//						X:       proto.Int(i),
//						Y:       proto.Int(j),
//						Channel: proto.Int(k),
//						Value:   proto.Float64(f.At(i, j, k)),
//					}
//					pixels = append(pixels, pixel)
//				}
//			}
//		}
//		return &Multi{
//			Width:       proto.Int(f.Width),
//			Height:      proto.Int(f.Height),
//			NumChannels: proto.Int(f.Channels),
//			Pixels:      pixels,
//		}
//	}

//	func multiFromProto(msg *Multi) *rimg64.Multi {
//		var (
//			width    = int(msg.GetWidth())
//			height   = int(msg.GetHeight())
//			channels = int(msg.GetNumChannels())
//		)
//		f := rimg64.NewMulti(width, height, channels)
//		n := width * height * channels
//		if len(msg.X) != n || len(msg.Y) != n || len(msg.Channel) != n || len(msg.Value) != n {
//			panic(fmt.Sprintf("bad dimensions: x %d, y %d, channel %d, value %d",
//				len(msg.X), len(msg.Y), len(msg.Channel), len(msg.Value)),
//			)
//		}
//
//		for p, val := range msg.Value {
//			var (
//				i = int(msg.X[p])
//				j = int(msg.Y[p])
//				k = int(msg.Channel[p])
//			)
//			f.Set(i, j, k, val)
//		}
//		return f
//	}

func multiFromProto(msg *Multi) *rimg64.Multi {
	var (
		width    = int(msg.GetWidth())
		height   = int(msg.GetHeight())
		channels = int(msg.GetNumChannels())
	)
	f := rimg64.NewMulti(width, height, channels)
	var (
		ei = int(msg.GetXStride())
		ej = int(msg.GetYStride())
		ek = int(msg.GetChannelStride())
	)
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				f.Set(i, j, k, msg.Elem[i*ei+j*ej+k*ek])
			}
		}
	}
	return f
}
