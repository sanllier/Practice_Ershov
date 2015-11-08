#ifndef __HELPERS__
#define __HELPERS__

//---------------------------------------------------------------------------------

#include <string>

#include "mpi.h"

using namespace std;

//---------------------------------------------------------------------------------

static const int MASTER = 0;

//---------------------------------------------------------------------------------

class MPIException {
	int m_errorCode;
	string m_callLine;

public:
	MPIException(int code, const string& call)
		: m_errorCode(code)
		, m_callLine(call) {}

	int code() const { return m_errorCode; }
	string call() const { return m_callLine; }

	string formattedString() const {
		return "--> MPI error occured\n        return code: " \
		       + to_string(m_errorCode) + "\n        call: " + m_callLine;
	}
};

//---------------------------------------------------------------------------------

#define MPICHECK(_x_) \
	{ \
		int retCode = _x_; \
		if (retCode != 0) throw MPIException(retCode, #_x_); \
	}

//---------------------------------------------------------------------------------

#endif
