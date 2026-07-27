#ifndef __CTYPE_H
#define __CTYPE_H

// <ctype.h> — pure guest C.
static int isdigit(int c) { return c >= '0' && c <= '9'; }
static int isspace(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
static int isalpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int isalnum(int c) { return isalpha(c) || isdigit(c); }
static int isupper(int c) { return c >= 'A' && c <= 'Z'; }
static int islower(int c) { return c >= 'a' && c <= 'z'; }
static int isxdigit(int c) { return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
static int isprint(int c) { return c >= 0x20 && c < 0x7f; }
static int ispunct(int c) { return isprint(c) && c != ' ' && !isalnum(c); }
static int iscntrl(int c) { return c < 0x20 || c == 0x7f; }
static int toupper(int c) { return islower(c) ? c - 'a' + 'A' : c; }
static int tolower(int c) { return isupper(c) ? c - 'A' + 'a' : c; }

#endif
