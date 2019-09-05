import nextprod


def testNexpow():
  assert nextprod.nextpow(2, 7) == 8
  assert nextprod.nextpow(2, 9) == 16
  assert nextprod.nextpow(5, 20) == 25
  assert nextprod.nextpow(4, 16) == 16


def testNextprod():
  assert nextprod.nextprod([2, 3], 105) == 108