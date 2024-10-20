---
layout: page
title: Schema Language
---

# Schema Language

Like Protocol Buffers and Thrift (but unlike JSON or MessagePack), Cap'n Proto messages are
strongly-typed and not self-describing. You must define your message structure in a special
language, then invoke the Cap'n Proto compiler (`capnp compile`) to generate source code to
manipulate that message type in your desired language.

For example:

{% highlight capnp %}
@0xdbb9ad1f14bf0b36;  # unique file ID, generated by `capnp id`

struct Person {
  name @0 :Text;
  birthdate @3 :Date;

  email @1 :Text;
  phones @2 :List(PhoneNumber);

  struct PhoneNumber {
    number @0 :Text;
    type @1 :Type;

    enum Type {
      mobile @0;
      home @1;
      work @2;
    }
  }
}

struct Date {
  year @0 :Int16;
  month @1 :UInt8;
  day @2 :UInt8;
}
{% endhighlight %}

Some notes:

* Types come after names. The name is by far the most important thing to see, especially when
  quickly skimming, so we put it up front where it is most visible.  Sorry, C got it wrong.
* The `@N` annotations show how the protocol evolved over time, so that the system can make sure
  to maintain compatibility with older versions. Fields (and enumerants, and interface methods)
  must be numbered consecutively starting from zero in the order in which they were added. In this
  example, it looks like the `birthdate` field was added to the `Person` structure recently -- its
  number is higher than the `email` and `phones` fields. Unlike Protobufs, you cannot skip numbers
  when defining fields -- but there was never any reason to do so anyway.

## Language Reference

### Comments

Comments are indicated by hash signs and extend to the end of the line:

{% highlight capnp %}
# This is a comment.
{% endhighlight %}

Comments meant as documentation should appear _after_ the declaration, either on the same line, or
on a subsequent line. Doc comments for aggregate definitions should appear on the line after the
opening brace.

{% highlight capnp %}
struct Date {
  # A standard Gregorian calendar date.

  year @0 :Int16;
  # The year.  Must include the century.
  # Negative value indicates BC.

  month @1 :UInt8;   # Month number, 1-12.
  day @2 :UInt8;     # Day number, 1-30.
}
{% endhighlight %}

Placing the comment _after_ the declaration rather than before makes the code more readable,
especially when doc comments grow long. You almost always need to see the declaration before you
can start reading the comment.

### Built-in Types

The following types are automatically defined:

* **Void:** `Void`
* **Boolean:** `Bool`
* **Integers:** `Int8`, `Int16`, `Int32`, `Int64`
* **Unsigned integers:** `UInt8`, `UInt16`, `UInt32`, `UInt64`
* **Floating-point:** `Float32`, `Float64`
* **Blobs:** `Text`, `Data`
* **Lists:** `List(T)`

Notes:

* The `Void` type has exactly one possible value, and thus can be encoded in zero bits. It is
  rarely used, but can be useful as a union member.
* `Text` is always UTF-8 encoded and NUL-terminated.
* `Data` is a completely arbitrary sequence of bytes.
* `List` is a parameterized type, where the parameter is the element type. For example,
  `List(Int32)`, `List(Person)`, and `List(List(Text))` are all valid.

### Structs

A struct has a set of named, typed fields, numbered consecutively starting from zero.

{% highlight capnp %}
struct Person {
  name @0 :Text;
  email @1 :Text;
}
{% endhighlight %}

Fields can have default values:

{% highlight capnp %}
foo @0 :Int32 = 123;
bar @1 :Text = "blah";
baz @2 :List(Bool) = [ true, false, false, true ];
qux @3 :Person = (name = "Bob", email = "bob@example.com");
corge @4 :Void = void;
grault @5 :Data = 0x"a1 40 33";
{% endhighlight %}

### Unions

A union is two or more fields of a struct which are stored in the same location. Only one of
these fields can be set at a time, and a separate tag is maintained to track which one is
currently set. Unlike in C, unions are not types, they are simply properties of fields, therefore
union declarations do not look like types.

{% highlight capnp %}
struct Person {
  # ...

  employment :union {
    unemployed @4 :Void;
    employer @5 :Company;
    school @6 :School;
    selfEmployed @7 :Void;
    # We assume that a person is only one of these.
  }
}
{% endhighlight %}

Additionally, unions can be unnamed.  Each struct can contain no more than one unnamed union.  Use
unnamed unions in cases where you would struggle to think of an appropriate name for the union,
because the union represents the main body of the struct.

{% highlight capnp %}
struct Shape {
  area @0 :Float64;

  union {
    circle @1 :Float64;      # radius
    square @2 :Float64;      # width
  }
}
{% endhighlight %}

Notes:

* Unions members are numbered in the same number space as fields of the containing struct.
  Remember that the purpose of the numbers is to indicate the evolution order of the
  struct. The system needs to know when the union fields were declared relative to the non-union
  fields.

* Notice that we used the "useless" `Void` type here. We don't have any extra information to store
  for the `unemployed` or `selfEmployed` cases, but we still want the union to distinguish these
  states from others.

* By default, when a struct is initialized, the lowest-numbered field in the union is "set".  If
  you do not want any field set by default, simply declare a field called "unset" and make it the
  lowest-numbered field.

* You can move an existing field into a new union without breaking compatibility with existing
  data, as long as all of the other fields in the union are new.  Since the existing field is
  necessarily the lowest-numbered in the union, it will be the union's default field.

**Wait, why aren't unions first-class types?**

Requiring unions to be declared inside a struct, rather than living as free-standing types, has
some important advantages:

* If unions were first-class types, then union members would clearly have to be numbered separately
  from the containing type's fields.  This means that the compiler, when deciding how to position
  the union in its containing struct, would have to conservatively assume that any kind of new
  field might be added to the union in the future.  To support this, all unions would have to
  be allocated as separate objects embedded by pointer, wasting space.

* A free-standing union would be a liability for protocol evolution, because no additional data
  can be attached to it later on.  Consider, for example, a type which represents a parser token.
  This type is naturally a union: it may be a keyword, identifier, numeric literal, quoted string,
  etc.  So the author defines it as a union, and the type is used widely.  Later on, the developer
  wants to attach information to the token indicating its line and column number in the source
  file.  Unfortunately, this is impossible without updating all users of the type, because the new
  information ought to apply to _all_ token instances, not just specific members of the union.  On
  the other hand, if unions must be embedded within structs, it is always possible to add new
  fields to the struct later on.

* When evolving a protocol it is common to discover that some existing field really should have
  been enclosed in a union, because new fields being added are mutually exclusive with it.  With
  Cap'n Proto's unions, it is actually possible to "retroactively unionize" such a field without
  changing its layout.  This allows you to continue being able to read old data without wasting
  space when writing new data.  This is only possible when unions are declared within their
  containing struct.

Cap'n Proto's unconventional approach to unions provides these advantages without any real down
side:  where you would conventionally define a free-standing union type, in Cap'n Proto you
may simply define a struct type that contains only that union (probably unnamed), and you have
achieved the same effect.  Thus, aside from being slightly unintuitive, it is strictly superior.

### Groups

A group is a set of fields that are encapsulated in their own scope.

{% highlight capnp %}
struct Person {
  # ...

  # Note:  This is a terrible way to use groups, and meant
  #   only to demonstrate the syntax.
  address :group {
    houseNumber @8 :UInt32;
    street @9 :Text;
    city @10 :Text;
    country @11 :Text;
  }
}
{% endhighlight %}

Interface-wise, the above group behaves as if you had defined a nested struct called `Address` and
then a field `address :Address`.  However, a group is _not_ a separate object from its containing
struct: the fields are numbered in the same space as the containing struct's fields, and are laid
out exactly the same as if they hadn't been grouped at all.  Essentially, a group is just a
namespace.

Groups on their own (as in the above example) are useless, almost as much so as the `Void` type.
They become interesting when used together with unions.

{% highlight capnp %}
struct Shape {
  area @0 :Float64;

  union {
    circle :group {
      radius @1 :Float64;
    }
    rectangle :group {
      width @2 :Float64;
      height @3 :Float64;
    }
  }
}
{% endhighlight %}

There are two main reason to use groups with unions:

1. They are often more self-documenting.  Notice that `radius` is now a member of `circle`, so
   we don't need a comment to explain that the value of `circle` is its radius.
2. You can add additional members later on, without breaking compatibility.  Notice how we upgraded
   `square` to `rectangle` above, adding a `height` field.  This definition is actually
   wire-compatible with the previous version of the `Shape` example from the "union" section
   (aside from the fact that `height` will always be zero when reading old data -- hey, it's not
   a perfect example).  In real-world use, it is common to realize after the fact that you need to
   add some information to a struct that only applies when one particular union field is set.
   Without the ability to upgrade to a group, you would have to define the new field separately,
   and have it waste space when not relevant.

Note that a named union is actually exactly equivalent to a named group containing an unnamed
union.

**Wait, weren't groups considered a misfeature in Protobufs?  Why did you do this again?**

They are useful in unions, which Protobufs did not have.  Meanwhile, you cannot have a "repeated
group" in Cap'n Proto, which was the case that got into the most trouble with Protobufs.

### Dynamically-typed Fields

A struct may have a field with type `AnyPointer`.  This field's value can be of any pointer type --
i.e. any struct, interface, list, or blob.  This is essentially like a `void*` in C.

See also [generics](#generic-types).

### Enums

An enum is a type with a small finite set of symbolic values.

{% highlight capnp %}
enum Rfc3092Variable {
  foo @0;
  bar @1;
  baz @2;
  qux @3;
  # ...
}
{% endhighlight %}

Like fields, enumerants must be numbered sequentially starting from zero. In languages where
enums have numeric values, these numbers will be used, but in general Cap'n Proto enums should not
be considered numeric.

### Interfaces

An interface has a collection of methods, each of which takes some parameters and return some
results.  Like struct fields, methods are numbered.  Interfaces support inheritance, including
multiple inheritance.

{% highlight capnp %}
interface Node {
  isDirectory @0 () -> (result :Bool);
}

interface Directory extends(Node) {
  list @0 () -> (list :List(Entry));
  struct Entry {
    name @0 :Text;
    node @1 :Node;
  }

  create @1 (name :Text) -> (file :File);
  mkdir @2 (name :Text) -> (directory :Directory);
  open @3 (name :Text) -> (node :Node);
  delete @4 (name :Text);
  link @5 (name :Text, node :Node);
}

interface File extends(Node) {
  size @0 () -> (size :UInt64);
  read @1 (startAt :UInt64 = 0, amount :UInt64 = 0xffffffffffffffff)
       -> (data :Data);
  # Default params = read entire file.

  write @2 (startAt :UInt64, data :Data);
  truncate @3 (size :UInt64);
}
{% endhighlight %}

Notice something interesting here: `Node`, `Directory`, and `File` are interfaces, but several
methods take these types as parameters or return them as results.  `Directory.Entry` is a struct,
but it contains a `Node`, which is an interface.  Structs (and primitive types) are passed over RPC
by value, but interfaces are passed by reference. So when `Directory.list` is called remotely, the
content of a `List(Entry)` (including the text of each `name`) is transmitted back, but for the
`node` field, only a reference to some remote `Node` object is sent.

When an address of an object is transmitted, the RPC system automatically manages making sure that
the recipient gets permission to call the addressed object -- because if the recipient wasn't
meant to have access, the sender shouldn't have sent the reference in the first place. This makes
it very easy to develop secure protocols with Cap'n Proto -- you almost don't need to think about
access control at all. This feature is what makes Cap'n Proto a "capability-based" RPC system -- a
reference to an object inherently represents a "capability" to access it.

### Generic Types

A struct or interface type may be parameterized, making it "generic". For example, this is useful
for defining type-safe containers:

{% highlight capnp %}
struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}

struct People {
  byName @0 :Map(Text, Person);
  # Maps names to Person instances.
}
{% endhighlight %}

Cap'n Proto generics work very similarly to Java generics or C++ templates. Some notes:

* Only pointer types (structs, lists, blobs, and interfaces) can be used as generic parameters,
  much like in Java. This is a pragmatic limitation: allowing parameters to have non-pointer types
  would mean that different parameterizations of a struct could have completely different layouts,
  which would excessively complicate the Cap'n Proto implementation.

* A type declaration nested inside a generic type may use the type parameters of the outer type,
  as you can see in the example above. This differs from Java, but matches C++. If you want to
  refer to a nested type from outside the outer type, you must specify the parameters on the outer
  type, not the inner. For example, `Map(Text, Person).Entry` is a valid type;
  `Map.Entry(Text, Person)` is NOT valid. (Of course, an inner type may declare additional generic
  parameters.)

* If you refer to a generic type but omit its parameters (e.g. declare a field of type `Map` rather
  than `Map(T, U)`), it is as if you specified `AnyPointer` for each parameter. Note that such
  a type is wire-compatible with any specific parameterization, so long as you interpret the
  `AnyPointer`s as the correct type at runtime.

* Relatedly, it is safe to cast an generic interface of a specific parameterization to a generic
  interface where all parameters are `AnyPointer` and vice versa, as long as the `AnyPointer`s are
  treated as the correct type at runtime. This means that e.g. you can implement a server in a
  generic way that is correct for all parameterizations but call it from clients using a specific
  parameterization.

* The encoding of a generic type is exactly the same as the encoding of a type produced by
  substituting the type parameters manually. For example, `Map(Text, Person)` is encoded exactly
  the same as:

  <div>{% highlight capnp %}
  struct PersonMap {
    # Encoded the same as Map(Text, Person).
    entries @0 :List(Entry);
    struct Entry {
      key @0 :Text;
      value @1 :Person;
    }
  }
  {% endhighlight %}
  </div>

  Therefore, it is possible to upgrade non-generic types to generic types while retaining
  backwards-compatibility.

* Similarly, a generic interface's protocol is exactly the same as the interface obtained by
  manually substituting the generic parameters.

### Generic Methods

Interface methods may also have "implicit" generic parameters that apply to a particular method
call. This commonly applies to "factory" methods. For example:

{% highlight capnp %}
interface Assignable(T) {
  # A generic interface, with non-generic methods.
  get @0 () -> (value :T);
  set @1 (value :T) -> ();
}

interface AssignableFactory {
  newAssignable @0 [T] (initialValue :T)
      -> (assignable :Assignable(T));
  # A generic method.
}
{% endhighlight %}

Here, the method `newAssignable()` is generic. The return type of the method depends on the input
type.

Ideally, calls to a generic method should not have to explicitly specify the method's type
parameters, because they should be inferred from the types of the method's regular parameters.
However, this may not always be possible; it depends on the programming language and API details.

Note that if a method's generic parameter is used only in its returns, not its parameters, then
this implies that the returned value is appropriate for any parameterization. For example:

{% highlight capnp %}
newUnsetAssignable @1 [T] () -> (assignable :Assignable(T));
# Create a new assignable. `get()` on the returned object will
# throw an exception until `set()` has been called at least once.
{% endhighlight %}

Because of the way this method is designed, the returned `Assignable` is initially valid for any
`T`. Effectively, it doesn't take on a type until the first time `set()` is called, and then `T`
retroactively becomes the type of value passed to `set()`.

In contrast, if it's the case that the returned type is unknown, then you should NOT declare it
as generic. Instead, use `AnyPointer`, or omit a type's parameters (since they default to
`AnyPointer`). For example:

{% highlight capnp %}
getNamedAssignable @2 (name :Text) -> (assignable :Assignable);
# Get the `Assignable` with the given name. It is the
# responsibility of the caller to keep track of the type of each
# named `Assignable` and cast the returned object appropriately.
{% endhighlight %}

Here, we omitted the parameters to `Assignable` in the return type, because the returned object
has a specific type parameterization but it is not locally knowable.

### Constants

You can define constants in Cap'n Proto.  These don't affect what is sent on the wire, but they
will be included in the generated code, and can be [evaluated using the `capnp`
tool](capnp-tool.html#evaluating-constants).

{% highlight capnp %}
const pi :Float32 = 3.14159;
const bob :Person = (name = "Bob", email = "bob@example.com");
const secret :Data = 0x"9f98739c2b53835e 6720a00907abd42f";
{% endhighlight %}

Additionally, you may refer to a constant inside another value (e.g. another constant, or a default
value of a field).

{% highlight capnp %}
const foo :Int32 = 123;
const bar :Text = "Hello";
const baz :SomeStruct = (id = .foo, message = .bar);
{% endhighlight %}

Note that when substituting a constant into another value, the constant's name must be qualified
with its scope.  E.g. if a constant `qux` is declared nested in a type `Corge`, it would need to
be referenced as `Corge.qux` rather than just `qux`, even when used within the `Corge` scope.
Constants declared at the top-level scope are prefixed just with `.`.  This rule helps to make it
clear that the name refers to a user-defined constant, rather than a literal value (like `true` or
`inf`) or an enum value.

### Nesting, Scope, and Aliases

You can nest constant, alias, and type definitions inside structs and interfaces (but not enums).
This has no effect on any definition involved except to define the scope of its name. So in Java
terms, inner classes are always "static". To name a nested type from another scope, separate the
path with `.`s.

{% highlight capnp %}
struct Foo {
  struct Bar {
    #...
  }
  bar @0 :Bar;
}

struct Baz {
  bar @0 :Foo.Bar;
}
{% endhighlight %}

If typing long scopes becomes cumbersome, you can use `using` to declare an alias.

{% highlight capnp %}
struct Qux {
  using Foo.Bar;
  bar @0 :Bar;
}

struct Corge {
  using T = Foo.Bar;
  bar @0 :T;
}
{% endhighlight %}

### Imports

An `import` expression names the scope of some other file:

{% highlight capnp %}
struct Foo {
  # Use type "Baz" defined in bar.capnp.
  baz @0 :import "bar.capnp".Baz;
}
{% endhighlight %}

Of course, typically it's more readable to define an alias:

{% highlight capnp %}
using Bar = import "bar.capnp";

struct Foo {
  # Use type "Baz" defined in bar.capnp.
  baz @0 :Bar.Baz;
}
{% endhighlight %}

Or even:

{% highlight capnp %}
using import "bar.capnp".Baz;

struct Foo {
  baz @0 :Baz;
}
{% endhighlight %}

The above imports specify relative paths.  If the path begins with a `/`, it is absolute -- in
this case, the `capnp` tool searches for the file in each of the search path directories specified
with `-I`.

### Annotations

Sometimes you want to attach extra information to parts of your protocol that isn't part of the
Cap'n Proto language.  This information might control details of a particular code generator, or
you might even read it at run time to assist in some kind of dynamic message processing.  For
example, you might create a field annotation which means "hide from the public", and when you send
a message to an external user, you might invoke some code first that iterates over your message and
removes all of these hidden fields.

You may declare annotations and use them like so:

{% highlight capnp %}
# Declare an annotation 'foo' which applies to struct and enum types.
annotation foo(struct, enum) :Text;

# Apply 'foo' to to MyType.
struct MyType $foo("bar") {
  # ...
}
{% endhighlight %}

The possible targets for an annotation are: `file`, `struct`, `field`, `union`, `enum`, `enumerant`,
`interface`, `method`, `parameter`, `annotation`, `const`.  You may also specify `*` to cover them
all.

{% highlight capnp %}
# 'baz' can annotate anything!
annotation baz(*) :Int32;

$baz(1);  # Annotate the file.

struct MyStruct $baz(2) {
  myField @0 :Text = "default" $baz(3);
  myUnion :union $baz(4) {
    # ...
  }
}

enum MyEnum $baz(5) {
  myEnumerant @0 $baz(6);
}

interface MyInterface $baz(7) {
  myMethod @0 (myParam :Text $baz(9)) -> () $baz(8);
}

annotation myAnnotation(struct) :Int32 $baz(10);
const myConst :Int32 = 123 $baz(11);
{% endhighlight %}

`Void` annotations can omit the value.  Struct-typed annotations are also allowed.  Tip:  If
you want an annotation to have a default value, declare it as a struct with a single field with
a default value.

{% highlight capnp %}
annotation qux(struct, field) :Void;

struct MyStruct $qux {
  string @0 :Text $qux;
  number @1 :Int32 $qux;
}

annotation corge(file) :MyStruct;

$corge(string = "hello", number = 123);

struct Grault {
  value @0 :Int32 = 123;
}

annotation grault(file) :Grault;

$grault();  # value defaults to 123
$grault(value = 456);
{% endhighlight %}

### Unique IDs

A Cap'n Proto file must have a unique 64-bit ID, and each type and annotation defined therein may
also have an ID.  Use `capnp id` to generate a new ID randomly.  ID specifications begin with `@`:

{% highlight capnp %}
# file ID
@0xdbb9ad1f14bf0b36;

struct Foo @0x8db435604d0d3723 {
  # ...
}

enum Bar @0xb400f69b5334aab3 {
  # ...
}

interface Baz @0xf7141baba3c12691 {
  # ...
}

annotation qux @0xf8a1bedf44c89f00 (field) :Text;
{% endhighlight %}

If you omit the ID for a type or annotation, one will be assigned automatically.  This default
ID is derived by taking the first 8 bytes of the MD5 hash of the parent scope's ID concatenated
with the declaration's name (where the "parent scope" is the file for top-level declarations, or
the outer type for nested declarations).  You can see the automatically-generated IDs by "compiling"
your file with the `-ocapnp` flag, which echos the schema back to the terminal annotated with
extra information, e.g. `capnp compile -ocapnp myschema.capnp`.  In general, you would only specify
an explicit ID for a declaration if that declaration has been renamed or moved and you want the ID
to stay the same for backwards-compatibility.

IDs exist to provide a relatively short yet unambiguous way to refer to a type or annotation from
another context.  They may be used for representing schemas, for tagging dynamically-typed fields,
etc.  Most languages prefer instead to define a symbolic global namespace e.g. full of "packages",
but this would have some important disadvantages in the context of Cap'n Proto:

* Programmers often feel the need to change symbolic names and organization in order to make their
  code cleaner, but the renamed code should still work with existing encoded data.
* It's easy for symbolic names to collide, and these collisions could be hard to detect in a large
  distributed system with many different binaries using different versions of protocols.
* Fully-qualified type names may be large and waste space when transmitted on the wire.

Note that IDs are 64-bit (actually, 63-bit, as the first bit is always 1).  Random collisions
are possible, but unlikely -- there would have to be on the order of a billion types before this
becomes a real concern.  Collisions from misuse (e.g. copying an example without changing the ID)
are much more likely.

## Evolving Your Protocol

A protocol can be changed in the following ways without breaking backwards-compatibility, and
without changing the [canonical](encoding.html#canonicalization) encoding of a message:

* New types, constants, and aliases can be added anywhere, since they obviously don't affect the
  encoding of any existing type.

* New fields, enumerants, and methods may be added to structs, enums, and interfaces, respectively,
  as long as each new member's number is larger than all previous members.  Similarly, new fields
  may be added to existing groups and unions.

* New parameters may be added to a method.  The new parameters must be added to the end of the
  parameter list and must have default values.

* Members can be re-arranged in the source code, so long as their numbers stay the same.

* Any symbolic name can be changed, as long as the type ID / ordinal numbers stay the same.  Note
  that type declarations have an implicit ID generated based on their name and parent's ID, but
  you can use `capnp compile -ocapnp myschema.capnp` to find out what that number is, and then
  declare it explicitly after your rename.

* Type definitions can be moved to different scopes, as long as the type ID is declared
  explicitly.

* A field can be moved into a group or a union, as long as the group/union and all other fields
  within it are new.  In other words, a field can be replaced with a group or union containing an
  equivalent field and some new fields.

* A non-generic type can be made [generic](#generic-types), and new generic parameters may be
  added to an existing generic type. Other types used inside the body of the newly-generic type can
  be replaced with the new generic parameter so long as all existing users of the type are updated
  to bind that generic parameter to the type it replaced. For example:

  <div>{% highlight capnp %}
  struct Map {
    entries @0 :List(Entry);
    struct Entry {
      key @0 :Text;
      value @1 :Text;
    }
  }
  {% endhighlight %}
  </div>

  Can change to:

  <div>{% highlight capnp %}
  struct Map(Key, Value) {
    entries @0 :List(Entry);
    struct Entry {
      key @0 :Key;
      value @1 :Value;
    }
  }
  {% endhighlight %}
  </div>

  As long as all existing uses of `Map` are replaced with `Map(Text, Text)` (and any uses of
  `Map.Entry` are replaced with `Map(Text, Text).Entry`).

  (This rule applies analogously to generic methods.)

The following changes are backwards-compatible but may change the canonical encoding of a message.
Apps that rely on canonicalization (such as some cryptographic protocols) should avoid changes in
this list, but most apps can safely use them:

* A field of type `List(T)`, where `T` is a primitive type, blob, or list, may be changed to type
  `List(U)`, where `U` is a struct type whose `@0` field is of type `T`.  This rule is useful when
  you realize too late that you need to attach some extra data to each element of your list.
  Without this rule, you would be stuck defining parallel lists, which are ugly and error-prone.
  As a special exception to this rule, `List(Bool)` may **not** be upgraded to a list of structs,
  because implementing this for bit lists has proven unreasonably expensive.

Any change not listed above should be assumed NOT to be safe.  In particular:

* You cannot change a field, method, or enumerant's number.
* You cannot change a field or method parameter's type or default value.
* You cannot change a type's ID.
* You cannot change the name of a type that doesn't have an explicit ID, as the implicit ID is
  generated based in part on the type name.
* You cannot move a type to a different scope or file unless it has an explicit ID, as the implicit
  ID is based in part on the scope's ID.
* You cannot move an existing field into or out of an existing union, nor can you form a new union
  containing more than one existing field.

Also, these rules only apply to the Cap'n Proto native encoding.  It is sometimes useful to
transcode Cap'n Proto types to other formats, like JSON, which may have different rules (e.g.,
field names cannot change in JSON).
