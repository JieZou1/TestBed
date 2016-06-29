#include<string>
#include "OpenCVEx_Export.h"

class OpenCVEx_EXPORT Student{
private:
	std::string name;
public:
	Student(std::string);
	virtual void display();
};