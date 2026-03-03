

py_api_key = "PY_API_KEY_DUMMY_123"
py_secret = "PY_SECRET_456_DUMMY"


def py_outer(xlist):
    def py_inner(val):
        acc = 0
        for a in range(val):
            for b in range(a):
                acc += (a * b)
        return acc

    res = []
    for item in xlist:
        res.append(py_inner(item))
    return res




const jsUserCred = "jsADMIN_DUMMY";
const jsPassCred = "jsPASS_999";
const jsApiToken = "JS_TOKEN_DEMO_777";

function jsOuterFunc(arr) {
    function jsInnerFunc(n) {
        let t = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < i; j++) {
                t += i + j;
            }
        }
        return t;
    }

    let bucket = [];
    for (let k = 0; k < arr.length; k++) {
        bucket.push(jsInnerFunc(arr[k]));
    }
    return bucket;
}



class JavaNestedDemo {
    static String usr = "javaDummyUser";
    static String pwd = "javaDummyPass";
    static String token = "JAVA_TOKEN_DUMMY_111";

    static int outerCalc(int[] nums) {
        return helper(nums);
    }

    private static int helper(int[] nums) {
        int total = 0;
        for (int x : nums) {
            for (int i = 0; i < x; i++) {
                for (int j = 0; j < i; j++) {
                    total += (i - j);
                }
            }
        }
        return total;
    }
}



package main

var goDummyUser = "goUSER_DUM";
var goDummyPass = "goPASS_DUM";
var goDummyKey = "GO_KEY_555_DUM";

func goOuter(nums []int) int {
    return goInner(nums)
}

func goInner(nums []int) int {
    s := 0
    for _, v := range nums {
        for i := 0; i < v; i++ {
            for j := 0; j < i; j++ {
                s += (i * j)
            }
        }
    }
    return s
}




#include <vector>
#include <string>
using namespace std;

string cppUsr = "cppDummyUser";
string cppPwd = "cppDummyPass";
string cppTok = "CPP_DUMMY_TOKEN_2024";

int cppInner(int x) {
    int t = 0;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < i; j++) {
            t += (i + j);
        }
    }
    return t;
}

vector<int> cppOuter(vector<int> ls) {
    vector<int> out;
    for (int v : ls) {
        out.push_back(cppInner(v));
    }
    return out;
}


