import pytest
from my_functions.build_features import map_fsize


class TestMapFsize(object):
    def test_on_family_size(self):
     
        test = {1:"single", 2:"small", 3:"medium", 4:"medium", 5:"large", 6:"large", 200:"large"}

        for key, value in test.items(): 

            actual = map_fsize(key)

            expected = value        

            message = ("map_fsize(1) "
                        "returned {0} instead "
                        "of {1}".format(actual, expected)
                        ) 
        assert actual is expected, message      









