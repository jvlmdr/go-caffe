# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)



DESCRIPTOR = descriptor.FileDescriptor(
  name='image.proto',
  package='caffe',
  serialized_pb='\n\x0bimage.proto\x12\x05\x63\x61\x66\x66\x65\"\x86\x01\n\x05Multi\x12\r\n\x05width\x18\x01 \x02(\x05\x12\x0e\n\x06height\x18\x02 \x02(\x05\x12\x14\n\x0cnum_channels\x18\x03 \x02(\x05\x12\x10\n\x08x_stride\x18\t \x02(\x05\x12\x10\n\x08y_stride\x18\n \x02(\x05\x12\x16\n\x0e\x63hannel_stride\x18\x0b \x02(\x05\x12\x0c\n\x04\x65lem\x18\x0c \x03(\x01')




_MULTI = descriptor.Descriptor(
  name='Multi',
  full_name='caffe.Multi',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='width', full_name='caffe.Multi.width', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='height', full_name='caffe.Multi.height', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='num_channels', full_name='caffe.Multi.num_channels', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='x_stride', full_name='caffe.Multi.x_stride', index=3,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='y_stride', full_name='caffe.Multi.y_stride', index=4,
      number=10, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='channel_stride', full_name='caffe.Multi.channel_stride', index=5,
      number=11, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='elem', full_name='caffe.Multi.elem', index=6,
      number=12, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=23,
  serialized_end=157,
)

DESCRIPTOR.message_types_by_name['Multi'] = _MULTI

class Multi(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _MULTI
  
  # @@protoc_insertion_point(class_scope:caffe.Multi)

# @@protoc_insertion_point(module_scope)