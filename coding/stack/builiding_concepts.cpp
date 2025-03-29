// #include <iostream>
// #include <string>
// #include <stack>
// using namespace std;

// int main()
// {
//     string s = "((())(";
//     stack<char> st;
//     for (int i = 0; i < s.size(); i++)
//     {
//         if (s[i] == '(' )
//         {
//             st.push(s[i]);
//         }
//         else
//         {
//             if (!st.empty() && st.top() == '(' )
//             {
//                 st.pop();
//             }else{
//                 st.push(s[i]);
//             }
//         }
//     }
//     if (st.empty())
//     {
//         cout << "valid" << endl;
//     }else{
//         cout << "invalid" << endl;
//     }

//     return 0;
// }

// basic concepts of the stack
// NGL

// #include <iostream>
// #include <vector>
// #include <stack>
// using namespace std;

// vector<int> nearest_element_to_left(vector<int> arr, int n){
//     vector<int> ans(n, -1); // default to -1
//     stack<int> st;

//     for(int i = 0; i< n; i++){
//        // 1 st
//         while(!st.empty() && st.top() <= arr[i]){
//             st.pop();
//         }
//         if(!st.empty()){
//             ans[i] = st.top();
//         }
//         st.push(arr[i]);
//     }

//     return ans;
// }

// int main(){
//     vector<int> arr = {8, 3, 7, 8, 9};
//     //  for each element there is the
//     // 8 3 5 7 8 9
//     // 8
//     //
//     vector<int> ans = nearest_element_to_left(arr, arr.size());

//     for (auto i : ans){
//         cout<< i<<endl;
//     }
//     return 0;
// }

// NGR

// #include <iostream>
// #include <vector>
// #include <stack>
// using namespace std;

// vector<int> ngr(vector<int> arr, int n){
//     vector<int> ans(n, -1);
//     stack<int> st;
//     for(int i = n -1; i >=0; i--){
//         while(!st.empty() && st.top() <= arr[i] ){
//             st.pop();
//         }
//         if(!st.empty()){
//             ans[i] = st.top();
//         }
//         st.push(arr[i]);
//     }
//     return ans;
// }

// int main(){
//     vector<int> arr = {2, 3, 4, 7, 2, 5, 2};

//     vector<int> ans = ngr(arr, arr.size());

//     // for(int i = ans.size() -1; i>=0; i--){
//     //     cout << ans[i] << endl;
//     // }

//     for (auto i: ans){
//         cout<<i<<endl;
//     }
//     return 0;
// }

// NSL
// #include <iostream>
// #include <vector>
// #include <stack>
// using namespace std;

// vector<int> nsl(vector<int> arr, int n)
// {
//     stack<int> st;
//     vector<int> ans(n, -1);
//     for (int i = 0; i < n; i++)
//     {
//         while (!st.empty() && arr[i] <= st.top())
//         {
//             st.pop();
//         }
//         if (!st.empty())
//         {
//             ans[i] = st.top();
//         }
//         st.push(arr[i]);
//     }
//     return ans;
// }
// int main()
// {
//     vector<int> arr = {1, 3, 4, 5, 1};
//     // 3 1 4
//     // -1 1
//     vector<int> ans = nsl(arr, arr.size());
//     for (auto i : ans)
//     {
//         cout << i << endl;
//     }
//     return 0;
// }




