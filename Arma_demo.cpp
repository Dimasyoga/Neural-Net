#include <iostream>
#include <armadillo>
#include <cmath>

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
		sum = 0;
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
		*it = pow((*it2 - *it1), 2);
		++it1;
		++it2;
	}
	return cost;
}

arma::mat quadraticLoss_d(arma::mat out, arma::mat goal) {
	arma::mat derivatif = out;
	arma::mat::iterator it_end = derivatif.end();
	arma::mat::iterator it1 = out.begin();
	arma::mat::iterator it2 = goal.begin();

	for (arma::mat::iterator it = derivatif.begin(); it != it_end; ++it) {
		*it = 0.5 * (*it2 - *it1);
		++it1;
		++it2;
	}
	return derivatif;
}

arma::mat calculateLayer(arma::mat input, arma::mat W, arma::mat bias) {

	return ((input * W) + bias);
}

void backProp(arma::mat error, arma::mat *W, char sel) {

}

int main()
{
	arma::arma_rng::set_seed_random();

	double error = 0;
	double learn_rate = 0.4;
	long long int iteration = 0;
	arma::mat inp(1, 3, arma::fill::randu);
	arma::mat l1(1, 3, arma::fill::zeros);
	arma::mat l2(1, 4, arma::fill::zeros);
	arma::mat out(1, 3, arma::fill::zeros);
	arma::mat goal(arma::size(out), arma::fill::randu);
	arma::mat loss(arma::size(out), arma::fill::zeros);

	arma::mat w1(inp.n_cols, l1.n_cols, arma::fill::randu);
	arma::mat w2(l1.n_cols, l2.n_cols, arma::fill::randu);
	arma::mat w3(l2.n_cols, out.n_cols, arma::fill::randu);

	arma::mat bias_l1(arma::size(l1), arma::fill::ones);
	arma::mat bias_l2(arma::size(l2), arma::fill::ones);
	arma::mat bias_out(arma::size(out), arma::fill::ones);

	std::cout << "input :" << std::endl << inp << std::endl;
	std::cout << "layer 1 :" << std::endl << relu(l1) << std::endl;
	std::cout << "layer 2 :" << std::endl << sigmoid(l2) << std::endl;
	std::cout << "output :" << std::endl << softmax(out) << std::endl;
	std::cout << std::endl;
	std::cout << "goal :" << std::endl << goal << std::endl;
	std::cout << std::endl;
	std::cout << "weight 1 :" << std::endl << w1 << std::endl;
	std::cout << "weight 2 :" << std::endl << w2 << std::endl;
	std::cout << "weight 3 :" << std::endl << w3 << std::endl;
	std::cout << "bias layer 1 :" << std::endl << bias_l1 << std::endl;
	std::cout << "bias layer 2 :" << std::endl << bias_l2 << std::endl;
	std::cout << "bias layer output :" << std::endl << bias_out << std::endl;

	system("pause");

	do {
		//forward propagation
		l1 = calculateLayer(inp, w1, bias_l1);
		l2 = calculateLayer(relu(l1), w2, bias_l2);
		out = calculateLayer(sigmoid(l2), w3, bias_out);

		loss = quadraticLoss(softmax(out), goal);
		
		arma::mat::iterator it_end = loss.end();

		for (arma::mat::iterator it = loss.begin(); it != it_end; ++it) {
			error += abs(*it);
		}

		error = error / loss.n_elem;

		std::cout << "Error : " << error << std::endl;

		//back propagation

		w3 += trans(sigmoid(l2)) * (quadraticLoss_d(softmax(out), goal) % softmax_d(out)) * learn_rate;
		bias_out += quadraticLoss_d(softmax(out), goal) % softmax_d(out) * learn_rate;

		w2 += (trans(relu(l1)) * sigmoid_d(l2)) * accu(quadraticLoss_d(softmax(out), goal) % softmax_d(out)) * learn_rate;
		bias_l2 += sigmoid_d(l2) * accu(quadraticLoss_d(softmax(out), goal) % softmax_d(out)) * learn_rate;

		w1 += (trans(inp) * relu_d(l1)) * accu(trans(sigmoid_d(l2)) * (quadraticLoss_d(softmax(out), goal) % softmax_d(out))) * learn_rate;
		bias_l1 += relu_d(l1) * accu(trans(sigmoid_d(l2)) * (quadraticLoss_d(softmax(out), goal) % softmax_d(out))) * learn_rate;
		iteration++;
	} while (error > 0.00000002);

	std::cout << "input :" << std::endl << inp << std::endl;
	std::cout << "layer 1 :" << std::endl << relu(l1) << std::endl;
	std::cout << "layer 2 :" << std::endl << sigmoid(l2) << std::endl;
	std::cout << "output :" << std::endl << softmax(out) << std::endl;
	std::cout << std::endl;
	std::cout << "goal :" << std::endl << goal << std::endl;
	std::cout << std::endl;
	std::cout << "weight 1 :" << std::endl << w1 << std::endl;
	std::cout << "weight 2 :" << std::endl << w2 << std::endl;
	std::cout << "weight 3 :" << std::endl << w3 << std::endl;
	std::cout << "bias layer 1 :" << std::endl << bias_l1 << std::endl;
	std::cout << "bias layer 2 :" << std::endl << bias_l2 << std::endl;
	std::cout << "bias layer output :" << std::endl << bias_out << std::endl;

	std::cout << "Error : " << error << std::endl;
	std::cout << "Iteration : " << iteration << std::endl;

	return 0;
}

