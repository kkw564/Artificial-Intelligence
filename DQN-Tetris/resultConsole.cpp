#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

typedef pair<int, int> pii;
string str;
vector<pii> vc;

bool comp(const pii &a, const pii &b) {
	return a.second > b.second;
}
int main() {
	freopen("../result1.txt", "r", stdin);

	int epi = 0, score = 0;
	for (int i = 0; i < 2389 + 1; i++) {
		getline(cin, str);
		
		epi = atoi(str.c_str() + 8);
		
		int pos = 0;
		while (1)
			if (str[pos++] == ',')
				break;

		while (1)
			if (str[pos++] == ':')
				break;
		pos++;
		score = atoi(str.c_str() + pos - 1);
		
		vc.push_back({ epi, score });
	}
	
	sort(vc.begin(), vc.end(), comp);

	for (int i = 0; i < vc.size(); i++)
		printf("epi :: %d score :: %d\n", vc[i].first, vc[i].second);

	return 0;
}