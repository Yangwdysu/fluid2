#ifndef MFD_LOG_H
#define MFD_LOG_H

#include <string>
#include <list>
#include <fstream>
#include <ctime>

namespace mfd {

	class Log
	{
	public:
		Log(){}
		~Log();

		/*!
			*	\brief	Types of logged messages.
			*/
		enum MessageType
		{
			DebugInfo,	//!< Message with some debug information.
			Info,		//!< Information to user.
			Warning,	//!< Warning information.
			Error,		//!< Error information while executing something.
			User		//!< User specific message.
		};

		/*!
			*	\brief	Logged message of type MessageType with some info.
			*/
		struct Message
		{
			MessageType type;
			std::string text;
			tm* when;
		};

		/*!
			*	\brief	Open file where to log the messages.
			*/
		static void SetOutput(const std::string& filename);

		/*!
			*	\brief	Get the filename of log.
			*/
		static const std::string& Output() { return outputFile; }

		/*!
			*	\brief	Add a new message to log.
			*	\param	type	Type of the new message.
			*	\param	text	Message.
			*	\remarks Message is directly passes to user reciever if one is set.
			*/
		static void Send(MessageType type, const std::string& text);

		/*!
			*	\brief	Get the list of all of the logged messages.
			*/
		static const std::list<Message>& Messages() { return messages; }

		/*!
			*	\brief	Get the last logged message.
			*/
		static const Message& LastMessage() { return messages.back(); }

		/*!
			*	\brief	Set user function to receive newly sent messages to logger.
			*/
		static void SetUserReceiver(void (*userFunc)(const Message&)) { receiver = userFunc; }

		/*!
			*	\brief	Set minimum level of message to be logged to file.
			*/
		static void SetLevel(MessageType level);

	private:

		static std::string outputFile;
		static std::ofstream outputStream;
		static std::list<Message> messages;
		static MessageType logLevel;
		static void (*receiver)(const Message&);
	};

	// simple debug macro
	#define LogDebug(DESC) Log::Send(Log::DebugInfo, DESC)

} // namespace mfd

#endif
