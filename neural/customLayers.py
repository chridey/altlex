import theano
import lasagne
import theano.tensor as T

class AverageOutputLayer(lasagne.layers.Layer):

    def __init__(self, incoming, mask, **kwargs):
        super(AverageOutputLayer, self).__init__(incoming, **kwargs)
        self.mask = mask

    def get_output_for(self, input, **kwargs):
        output_sums = T.sum(input * self.mask[:, :, None], axis=1)
        mask_sums = T.sum(self.mask, axis=1)
        return output_sums / mask_sums[:, None]

    def get_output_shape_for(self, input_shape):
        return (None, input_shape[-1])

class AltlexSliceLayer(lasagne.layers.Layer):

    def __init__(self, incoming, idxs, axis=1, **kwargs):
        super(AltlexSliceLayer, self).__init__(incoming, **kwargs)
        self.idxs = idxs
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        idx = self.idxs
        return input[T.arange(input.shape[0]), idx]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
