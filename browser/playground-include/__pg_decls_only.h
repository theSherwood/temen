/* Decls-only opt-in for a *program* translation unit (#1392).
 *
 * Seeded under `include/` and force-included (`-include __pg_decls_only.h`) when a program is
 * compiled as a unit that links against a prebuilt libc unit. Defining the macro before any header
 * is read makes the seeded libc headers expose prototypes only, so the program's compile emits no
 * libc bodies at all (~12x cheaper). The libc unit itself is compiled *without* this, so it is the
 * one place the bodies are instantiated.
 *
 * It exists as a header rather than a `-D` on the command line because the committed
 * `chibicc.temen` asset predates `-D` support; `-include` has been there all along.
 */
#define __PG_LIBC_DECLS_ONLY 1
