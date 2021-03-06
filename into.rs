use super::IntoIterator;

pub trait Collect: IntoIterator+Sized { fn collect<B>(self) -> B where B: std::iter::FromIterator<Self::Item> { self.into_iter().collect() } }
// impl Collect for IntoIterator !B
impl<I, F> Collect for Map<I, F> where Self:IntoIterator {}

pub struct Copied<I>(I);
impl<T:Copy+'t, I:IntoIterator<Item=&'t T>> IntoIterator for Copied<I> {
	type IntoIter = std::iter::Copied::<I::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::copied(self.0.into_iter()) }
}
impl<I:Clone> Clone for Copied<I> { fn clone(&self) -> Self { Copied(self.0.clone()) } }

pub trait IntoCopied : IntoIterator+Sized { fn copied(self) -> Copied<Self>; }
impl<T:Copy+'t, I:IntoIterator<Item=&'t T>> IntoCopied for I { fn copied(self) -> Copied<Self> { Copied(self) } }

pub trait Enumerate : IntoIterator+Sized { fn enumerate(self) -> std::iter::Enumerate<Self::IntoIter> { self.into_iter().enumerate() } }
impl<I:IntoIterator> Enumerate for I {}

pub struct Chain<A,B>{a: A, b: B}
impl<A:IntoIterator, B:IntoIterator<Item=A::Item>> IntoIterator for Chain<A, B> {
	type IntoIter = super::ChainIterator::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { super::IntoChain::chain(self.a.into_iter(), self.b.into_iter()) }
}
pub trait IntoChain : Sized { fn chain<B: IntoIterator+Sized>(self, b: B) -> Chain<Self, B> { Chain{a: self, b: b} } }
impl<A: IntoIterator> IntoChain for A {}

pub struct Zip<A,B>{pub a: A, pub b: B}
pub trait IntoZip : Sized { fn zip<B>(self, b: B) -> Zip<Self, B> { Zip{a: self, b} } }
impl<I> IntoZip for I {}
pub fn zip<A,B>(a: A, b: B) -> Zip<A, B> { IntoZip::zip(a, b) }
impl<A:IntoIterator, B:IntoIterator> IntoIterator for Zip<A, B> {
	type IntoIter = std::iter::Zip::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::zip(self.a.into_iter(), self.b.into_iter()) }
}
/*impl<A:crate::IntoExactSizeIterator, B:crate::IntoExactSizeIterator> IntoIterator for Zip<A, B> {
	type IntoIter = std::iter::Zip::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { let (a,b) = (self.a.into_iter(), self.b.into_iter()); assert!(self.0.len()==self.1.len()); Iterator::zip(a,b) }
}*/
impl<A, B> IntoIterator for &'t Zip<A, B> where &'t A:IntoIterator, &'t B:IntoIterator {
	type IntoIter = std::iter::Zip::<<&'t A as IntoIterator>::IntoIter,<&'t B as IntoIterator>::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::zip(self.a.into_iter(), self.b.into_iter()) }
}
#[macro_export] macro_rules! zip {
  ($first:expr $(,)*) => { $first };
	($first:expr, $second:expr $(,)*) => { {$crate::into::IntoZip::zip($crate::zip!($first), $second)} };
	($first:expr $( , $rest:expr )* $(,)* ) => { {use $crate::into::IntoZip; $crate::into::IntoMap::map($crate::zip!($first) $( .zip($rest) )*,$crate::zip!(@closure a => (a) $( , $rest )*) ) } };
	( @closure $p:pat => $tup:expr ) => { |$p| $tup };
	( @closure $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )* ) => { $crate::zip!(@closure ($p, b) => ( $($tup)*, b ) $( , $tail )*) };
}

#[derive(Clone, Copy)] pub struct Map<I,F>{iter: I, f: F}
pub trait IntoMap : IntoIterator+Sized { fn map<F:Fn<(Self::Item,)>>(self, f: F) -> Map<Self, F> { Map{iter: self, f} } }
impl<I:IntoIterator> IntoMap for I {}
impl<I:IntoIterator, F: Fn<(I::Item,)>> IntoIterator for Map<I, F> {
	type IntoIter = std::iter::Map::<I::IntoIter, F>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::map(self.iter.into_iter(), self.f) }
}

pub trait Find : IntoIterator+Sized { fn find<P:FnMut(&Self::Item)->bool>(self, predicate: P) -> Option<Self::Item> { self.into_iter().find(predicate) } }
impl<I:IntoIterator> Find for I {}

pub trait Filter : IntoIterator+Sized { fn filter<F:FnMut(&Self::Item)->bool>(self, f: F) -> std::iter::Filter<Self::IntoIter, F> { self.into_iter().filter(f) } }
impl<I:IntoIterator> Filter for I {}

pub trait FilterMap : IntoIterator+Sized { fn filter_map<U, F:FnMut(Self::Item)->Option<U>>(self, f: F) -> std::iter::FilterMap<Self::IntoIter, F> { self.into_iter().filter_map(f) } }
impl<I:IntoIterator> FilterMap for I {}

pub trait Fold : IntoIterator+Sized { fn fold<B, F:FnMut(B, Self::Item) -> B>(self, init: B, f: F) -> B { self.into_iter().fold(init, f) } }
impl<I:IntoIterator> Fold for I {}

pub trait Sum<T> { fn sum(self) -> T; }
impl<I:IntoIterator, T:std::iter::Sum<I::Item>> Sum<T> for I { fn sum(self) -> T { Iterator::sum(self.into_iter()) } }

#[cfg(feature="itertools")] pub trait Format : IntoIterator+Sized {
	fn format(self, sep: &str) -> itertools::Format<'_, Self::IntoIter> { itertools::Itertools::format(self.into_iter(), sep) }
	fn format_with<F: FnMut(Self::Item, &mut dyn FnMut(&dyn std::fmt::Display) -> std::fmt::Result) -> std::fmt::Result>(self, sep: &str, format: F) -> itertools::FormatWith<'_, Self::IntoIter, F> {
		itertools::Itertools::format_with(self.into_iter(), sep, format)
	}
}
#[cfg(feature="itertools")] impl<I:IntoIterator> Format for I {}
