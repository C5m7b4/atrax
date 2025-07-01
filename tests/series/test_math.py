import pytest
from datetime import datetime
from atrax import Series


# basic math
def test_add():
    s = Series([1,2,3])
    s1 = Series([1,2,3])

    r1 = s + 1
    assert r1.data == [2,3,4]

    r2 = s + s1
    assert r2.data == [2, 4, 6]

def test_add_different_lengths():
    s = Series([1,2,3])
    s1 = Series([1,2,3,4])

    with pytest.raises(ValueError):
        r1 = s + s1

def test_sub():
    s = Series([1,2,3])
    s1 = Series([1,2,3])  

    r1 = s - 1
    assert r1.data == [0, 1, 2] 

    r2 = s - s1
    assert r2.data == [0, 0, 0]

def test_mul():    
    s = Series([1,2,3])
    s1 = Series([1,2,3])  

    r1 = s1 * 2
    assert r1.data == [2, 4, 6]

    r2 = s * s1
    assert r2.data == [1, 4, 9]

def test_true_div():
    s = Series([1,4,6])
    s1 = Series([1,2,2])  

    r1 = s / 5
    assert r1.data == [.2, .8, 1.2]   

    r2 = s / s1
    assert r2.data == [1, 2, 3]      

def test_floor_div():
    s = Series([1,4,6])
    s1 = Series([1,2,2])  

    r1 = s // 5
    assert r1.data == [0, 0, 1]   

    r2 = s // s1
    assert r2.data == [1, 2, 3]    

def test_mod():
    s = Series([1,4,6])
    s1 = Series([1,2,2])       

    r1 = s % 2
    assert r1.data == [1, 0, 0]  

    r2 = s % s1
    assert r2.data == [0,0, 0] 

def test_pow():
    s = Series([1,4,6])
    s1 = Series([1,2,2])   

    r1 = s ** 2
    assert r1.data == [1, 16, 36]   

    r2 = s ** s1
    assert r2.data == [1, 16, 36]     

 # reverse math

def test_rev_add():
    s = Series([1,4,6])
    s1 = Series([1,2,2]) 

    r1 = 1 + s
    assert r1.data == [2, 5, 7] 

    r2 = s1 + s
    assert r2.data == [2, 6, 8] 

def test_rev_sub():
    s = Series([1,4,6])
    s1 = Series([1,2,2])     

    r1 = 1 - s
    assert r1.data == [0, -3, -5]

    r2 = s1 - s
    assert r2.data == [0, -2, -4]

def test_rev_mul():
    s = Series([1,4,6])
    s1 = Series([1,2,2]) 

    r1 = 2 * s
    assert r1.data == [2, 8, 12]    

    r2 = s1 * s
    assert  r2.data == [1, 8, 12] 

def test_rtrue_div():
    s = Series([1,6,12])
    s1 = Series([1,2,2])   

    r1 = 3 / s
    assert r1.data == [3.0, 0.5, .25] 

    r2 = s1 / s
    assert r2.data == [1, 0.3333333333333333, 0.16666666666666666]  

def test_rfloor_div():
    s = Series([1,6,12])
    s1 = Series([1,2,2])     

    r1 = 2 // s
    assert r1.data == [2, 0, 0] 

    r2 = s1 // s
    assert r2.data == [1, 0, 0]

def test_rmod():
    s = Series([1,5,13])
    s1 = Series([8,12,26])      

    r1 = 2 % s
    assert r1.data == [0, 2, 2]   

    r2 = s1 % s
    assert r2.data == [0, 2, 0]

def test_rpow():
    s = Series([1,2,3])
    s1 = Series([2,3,4])    

    r1 = 2 ** s
    assert r1.data == [2, 4, 8]

    r2 = s1 ** s
    assert r2.data == [2, 9, 64]
