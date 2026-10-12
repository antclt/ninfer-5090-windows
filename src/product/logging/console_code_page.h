#pragma once

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace ninfer::product {

// The console's output code page belongs to the console, not to this process: setting it changes it for
// every program attached to that console, and it outlives this one. Measured on this machine -- a child
// that set 65001 left its parent reading 65001 after it exited. So it is read first, set only when
// stdout is actually a console and the page differs, and restored on the way out.
//
// One copy for the three apps. cli, serve and perplexity each carried this class and this comment
// verbatim; the measurement above is the reason it exists at all, and a guard duplicated three times is
// a guard that gets fixed once and stays broken twice.
class ConsoleOutputCodePage {
public:
    ConsoleOutputCodePage() {
        const HANDLE out = ::GetStdHandle(STD_OUTPUT_HANDLE);
        if (out == nullptr || out == INVALID_HANDLE_VALUE || !::GetConsoleMode(out, &mode_)) {
            return; // redirected, or no console: nothing to change and nothing to restore
        }
        previous_ = ::GetConsoleOutputCP();
        if (previous_ != CP_UTF8 && ::SetConsoleOutputCP(CP_UTF8) != FALSE) { changed_ = true; }
    }
    ~ConsoleOutputCodePage() {
        if (changed_) { ::SetConsoleOutputCP(previous_); }
    }
    ConsoleOutputCodePage(const ConsoleOutputCodePage&)            = delete;
    ConsoleOutputCodePage& operator=(const ConsoleOutputCodePage&) = delete;

private:
    DWORD mode_    = 0;
    UINT previous_ = 0;
    bool changed_  = false;
};

} // namespace ninfer::product

#endif // _WIN32
