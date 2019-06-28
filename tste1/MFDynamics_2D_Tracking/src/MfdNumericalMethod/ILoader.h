#ifndef MFD_LOADER_H
#define MFD_LOADER_H

#include <string>


namespace mfd {

	class ISimulation;

	/*!
	 *	\class	Loader
	 *	\brief	Abstract class for reading input data.
	 *	\todo	create cpp file and on constructor setlocale(LC_ALL, "C");
	 */
	class ILoader
	{
	public:

		ILoader() {}
		virtual ~ILoader() {}

		/*!
		 *	\brief	Set the file to read.
		 */
		void SetInput(const std::string& inputPath)	{ path = inputPath; }

		/*!
		 *	\brief	Parse the file and return built simulation object.
		 *	\return	New Simulation object.
		 */
		virtual ISimulation* Read() = 0;

	protected:

		std::string path;

	};


}

#endif
