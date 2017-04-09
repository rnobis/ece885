from keras.layers import LSTM
from keras import backend as K
from keras.engine import InputSpec


class LSTM1(LSTM):
   def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]

        else:           
            x_i = K.dot(x * B_W[0], K.zeros((self.input_dim, self.output_dim))) + self.b_i
            x_f = K.dot(x * B_W[1], K.zeros((self.input_dim, self.output_dim))) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], K.zeros((self.input_dim, self.output_dim))) + self.b_o


        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + 1*K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + 1*K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]
        
   def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.U_i = self.inner_init((self.output_dim,self.output_dim),
                          name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))



        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))


        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))


        self.trainable_weights = [self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.U_f, self.b_f,
                                  self.U_o, self.b_o]


class LSTM2(LSTM):
   def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]

        else:           
            x_i = K.dot(x * B_W[0], K.zeros((self.input_dim, self.output_dim))) 
            x_f = K.dot(x * B_W[1], K.zeros((self.input_dim, self.output_dim))) 
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], K.zeros((self.input_dim, self.output_dim)))


        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + 1*K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + 1*K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]
        
   def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.U_i = self.inner_init((self.output_dim,self.output_dim),
                          name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))



        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))


        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))


        self.trainable_weights = [self.U_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.U_f,
                                  self.U_o]
                                  
                                  
class LSTM3(LSTM):
   def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]

        else:           
            x_i = K.dot(x * B_W[0], K.zeros((self.input_dim, self.output_dim))) + self.b_i
            x_f = K.dot(x * B_W[1], K.zeros((self.input_dim, self.output_dim))) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], K.zeros((self.input_dim, self.output_dim))) + self.b_o


        i = self.inner_activation(x_i)
        f = self.inner_activation(x_f)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o)

        h = o * self.activation(c)
        return h, [h, c]
        
   def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.U_i = self.inner_init((self.output_dim,self.output_dim),
                          name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))



        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))


        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))


        self.trainable_weights = [self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.b_f,
                                  self.b_o]


