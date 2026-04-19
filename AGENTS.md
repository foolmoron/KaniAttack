# AGENTS.md

## Platform

- Engine: LittleJS
- Language: TypeScript
- Source folder: src/
- Build folder: build/
- Reference docs: @littlejs.md contains a cheat sheet for LittleJS

## Coding Rules

- **TypeScript:** Use TypeScript for source files
- **Types:** Use interfaces for simple types especially inherited ones. Use algebraic types for more complicated ones including generics
- **Classes:** Avoid. Use more functional and imperative structures.
- **Comments:** Avoid. Only use them to explain _why_ something unintuitive is done, never _what_ it does.
- **Naming:** Use camelCase for naming all files and variables, PascalCase for globals/statics.
- **Formatting:** Don't concern yourself with formatting, Prettier will handle it on save.
- **npm commands:** Never offer to run `npm run dev` or `npm run typecheck` or anything like that, I will run it manually.
- **Braces:** Always use opening/closed braces for if/for/etc.
