extern crate ndarray;
extern crate rand;

use ndarray::{Array, Array1, Array2};
use rand::{thread_rng, Rng};

struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut rng = thread_rng();
        let num_layers = layer_sizes.len();
        let weights = (0..num_layers - 1)
            .map(|i| Array::random((layer_sizes[i + 1], layer_sizes[i]), &mut rng))
            .collect();
        let biases = (1..num_layers)
            .map(|i| Array::random(layer_sizes[i], &mut rng))
            .collect();
        NeuralNetwork { weights, biases }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn feedforward(&self, input: ArrayView1<f64>) -> Array1<f64> {
        let mut activation = input.to_owned();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            activation = w.dot(&activation) + b;
            activation.mapv_inplace(Self::sigmoid);
        }
        activation
    }
}

fn main() {
    let nn = NeuralNetwork::new(&[2, 3, 1]);
    let input = Array::from_vec(vec![1.0, 0.0]);
    let output = nn.feedforward(input.view());
    println!("Input: {:?}", input);
    println!("Output: {:?}", output);
}
