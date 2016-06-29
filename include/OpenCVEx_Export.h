#ifndef _OpenCVExExport_H_
#define _OpenCVExExport_H_

/* Cmake will define OpenCVEx_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define MyLibrary_EXPORTS when
building a DLL on windows.
*/
// We are using the Visual Studio Compiler and building Shared libraries

#if defined (_WIN32) 
#if defined(OpenCVEx_EXPORTS)
#define  OpenCVEx_EXPORT __declspec(dllexport)
#else
#define  OpenCVEx_EXPORT __declspec(dllimport)
#endif /* OpenCVEx_EXPORTS */
#else /* defined (_WIN32) */
#define OpenCVEx_EXPORT
#endif

#endif /* _OpenCVExExport_H_ */
