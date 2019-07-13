#include <iostream>
#include <armadillo>
#include <cmath>
#include <string.h>

//double calcError(arma::mat x) {
//	arma::mat::iterator it_end = x.end();
//	double error = 0;
//	for (arma::mat::iterator it = x.begin(); it != it_end; ++it) {
//		error += pow(*it, 2);
//	}
//	return sqrt(error / float(size(x).n_cols * size(x).n_rows)) * 100.0;
//}
//
//double sigmoid(double x) {
//	return 1 / (1 + exp(-1 * x));
//}
//
//arma::mat calcSigmoid_d(arma::mat x) {
//	arma::mat z = x;
//	arma::mat::iterator it_end = z.end();
//	for (arma::mat::iterator it = z.begin(); it != it_end; ++it) {
//		*it = sigmoid(*it)*(1 - sigmoid(*it));
//	}
//
//	return z;
//}
//
//arma::mat calcWeight_delta(arma::mat error, arma::mat out, arma::mat inp) {
//	arma::mat out_delta = error % calcSigmoid_d(out);
//	arma::mat deltas = out_delta * inp.t();
//	return deltas;
//}
//
//arma::mat calcSigmoid(arma::mat x, arma::mat y) {
//	arma::mat z = x * y;
//	arma::mat::iterator it_end = z.end();
//	for (arma::mat::iterator it = z.begin(); it != it_end; ++it) {
//		*it = sigmoid(*it + 1);
//	}
//	return z;
//}

arma::mat relu(arma::mat input) {
	arma::mat out = input;
	arma::mat::iterator it_end = input.end();

	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		if (*it < 0) {
			*it = 0;
		}
	}
	return out;
}

arma::mat relu_d(arma::mat input) {
	arma::mat out = input;
	arma::mat::iterator it_end = input.end();

	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		if (*it < 0) {
			*it = 0;
		}
		else {
			*it = 1;
		}
	}
	return out;
}

arma::mat sigmoid(arma::mat input) {
	arma::mat out = input;
	arma::mat::iterator it_end = input.end();

	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		*it = 1 / (1 + exp(-1 * (*it)));
	}
	return out;
}

arma::mat sigmoid_d(arma::mat input) {
	arma::mat out = input;
	arma::mat::iterator it_end = input.end();

	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		*it = (1 / (1 + exp(-1 * (*it)))) * (1 - (1 / (1 + exp(-1 * (*it)))));
	}
	return out;
}

arma::mat softmax(arma::mat input) {
	arma::mat out = input;
	double sumExp = 0;
	arma::mat::iterator it_end = input.end();

	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		sumExp += exp(*it);
	}
	for (arma::mat::iterator it = input.begin(); it != it_end; ++it) {
		*it = *it / sumExp;
	}
	return out;
}

arma::mat softmax_d(arma::mat input) {
	arma::mat out = input;
	double sumExp = 0;
	arma::mat::iterator it_end = out.end();

	for (arma::mat::iterator it = out.begin(); it != it_end; ++it) {
		sumExp += exp(*it);
	}
	double sum;
	for (arma::mat::iterator it = out.begin(); it != it_end; ++it) {
		arma::mat::iterator it2_end = input.end();
		double sum = 0;
		for (arma::mat::iterator it2 = input.begin(); it2 != it2_end; ++it2) {
			if (*it2 != *it) {
				sum += *it2;
			}
		}
		*it = (*it * sum) / pow(sumExp, 2);
	}
	return out;
}

arma::mat quadraticLoss(arma::mat out, arma::mat goal) {
	
	arma::mat cost = out;
	arma::mat::iterator it_end = cost.end();
	arma::mat::iterator it1 = out.begin();
	arma::mat::iterator it2 = goal.begin();

	for (arma::mat::iterator it = cost.begin(); it != it_end; ++it) {
		*it = pow(2, (*it2 - *it));
		it1++;
		it2++;
	}
	return cost;
}

arma::mat quadraticLoss_d(arma::mat out, arma::mat goal) {
	arma::mat derivatif = out;
	arma::mat::iterator it_end = derivatif.end();
	arma::mat::iterator it1 = out.begin();
	arma::mat::iterator it2 = goal.begin();

	for (arma::mat::iterator it = derivatif.begin(); it != it_end; ++it) {
		++it1; ++it2;
		*it = 0.5 * (*it2 - *it1);
	}
	return derivatif;
}

arma::mat calculateLayer(arma::mat input, arma::mat W, char select) {
	arma::mat output(input.n_rows, W.n_cols, arma::fill::zeros);
	
	if (select == 'r') { //ReLu
		output = relu(input * W);
	}
	else if (select == 's') { //Sigmoid
		output = sigmoid(input * W);
	}
	else if (select == 'm') { //softmax
		output = softmax(input * W);
	}
	else {
		return NULL;
	}

	return output;
}

void backProp(arma::mat error, arma::mat *W, char sel) {

}

int main()
{
	float error = 0.0;
	arma::mat inp(1, 3, arma::fill::randu);
	arma::mat l1(1, 2, arma::fill::zeros);
	arma::mat l2(1, 3, arma::fill::zeros);
	arma::mat out(1, 2, arma::fill::zeros);
	arma::mat goal(out.size, arma::fill::randu);
	arma::mat loss(out.size, arma::fill::zeros);

	arma::mat w1(inp.n_cols, l1.n_cols, arma::fill::randu);
	arma::mat w2(l1.n_cols, l2.n_cols, arma::fill::randu);
	arma::mat w3(l2.n_cols, out.n_cols, arma::fill::randu);

	arma::mat bias_l1(l1.size, arma::fill::ones);
	arma::mat bias_l2(l2.size, arma::fill::ones);
	arma::mat bias_out(out.size, arma::fill::ones);

	do {
		//forward propagation
		l1 = calculateLayer(inp, w1, 'r') + bias_l1;
		l2 = calculateLayer(l1, w2, 's') + bias_l2;
		out = calculateLayer(l2, w3, 'm') + bias_out;

		loss = quadraticLoss(out, goal);
		
		arma::mat::iterator it_end = loss.end();

		for (arma::mat::iterator it = loss.begin(); it != it_end; ++it) {
			error += *it;
		}

		error = error / loss.n_cols;

		//back propagation




	} while (error > 0.02);

	return 0;
}

