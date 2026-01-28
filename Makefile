CFLAGS += -g -Wall -O3
CXXFLAGS += -g -Wall -O3

test.o: test.c test.h
	$(CC) -c -o $@ $(CFLAGS) $<

bch_test: bch_test.o bch_codec.o
	$(CXX) -o $@ $(CXXFLAGS) $^

test: bch_test
	./bch_test

clean:
	@rm -f bch_test *.o
