#ifndef MFD_SIMULATION_H
#define MFD_SIMULATION_H
#include <string>

namespace mfd {

	class ISimulation
	{
	public:
		ISimulation(void) {};
		virtual ~ISimulation(void) {};

		virtual bool Initialize( std::string in_filename ) = 0;

		virtual void Advance(float dt) = 0;
		virtual float GetTimeStep() = 0;
		virtual void Run() {};

		virtual void PreProcessing() {};
		virtual void PostProcessing() {};

		virtual void Invoke(unsigned char type, unsigned char key, int x, int y) {};

	public:
	};

}

#endif