def fft(poly):#poly is a list of coefficients of a polynomial
    """returns the evalutation of the polynomial at the n-th roots of unity"""
    if len(poly) == 1:
        return poly
    else:
        even = fft(poly[::2])
        odd = fft(poly[1::2])
        result = [0] * len(poly)
        for i in range(len(poly)//2):
            result[i] = even[i] + odd[i] * (1j)**i
            result[i+ len(poly)//2] = even[i] - odd[i] * (1j)**i
        return result

def multiplyPoly(poly1, poly2):
    """multiplies two polynomials using fft"""
    poly1 = poly1 + [0] * (len(poly2) - len(poly1))
    poly2 = poly2 + [0] * (len(poly1) - len(poly2))
    poly1 = fft(poly1+[0]*len(poly1))
    poly2 = fft(poly2+[0]*len(poly2))
    return [x*y for x, y in zip(poly1, poly2)]

poly1 = [1, 1]
poly2 = [1, 1]
result = [1, 2, 1]
print(multiplyPoly(poly1, poly2))
print(fft(result))
