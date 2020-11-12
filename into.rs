use super::IntoIterator;

pub trait Collect: IntoIterator+Sized { fn collect<B>(self) -> B where B: std::iter::FromIterator<Self::Item> { self.into_iter().collect() } }
// impl Collect for IntoIterator !B
impl<I, F> Collect for Map<I, F> where Self:IntoIterator {}

pub trait Enumerate : IntoIterator+Sized { fn enumerate(self) -> std::iter::Enumerate<Self::IntoIter> { self.into_iter().enumerate() } }
impl<I:IntoIterator> Enumerate for I {}

pub struct Chain<A,B>{a: A, b: B}
impl<A:IntoIterator, B:IntoIterator<Item=A::Item>> IntoIterator for Chain<A, B> {
	type IntoIter = super::Chain::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { super::IntoChain::chain(self.a.into_iter(), self.b.into_iter()) }
}

pub trait IntoChain<B:Sized> : Sized { type Output; fn chain(self, b: B) -> Self::Output; }
impl<A: IntoIterator, B: IntoIterator+Sized> IntoChain<B> for A {
	type Output = Chain<A, B>;
	fn chain(self, b: B) -> Self::Output { Chain{a: self, b: b} }
}

pub struct Zip<A,B>{pub a: A, pub b: B}
pub trait IntoZip : Sized { fn zip<B>(self, b: B) -> Zip<Self, B> { Zip{a: self, b} } }
impl<I> IntoZip for I {}
pub fn zip<A,B>(a: A, b: B) -> Zip<A, B> { IntoZip::zip(a, b) }
impl<A:IntoIterator, B:IntoIterator> IntoIterator for Zip<A, B> {
	type IntoIter = std::iter::Zip::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::zip(self.a.into_iter(), self.b.into_iter()) }
}
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

pub struct Map<I,F>{iter: I, f: F}
pub trait IntoMap : IntoIterator+Sized { fn map<F:Fn<(Self::Item,)>>(self, f: F) -> Map<Self, F> { Map{iter: self, f} } }
impl<I:IntoIterator> IntoMap for I {}
pub fn map<V:IntoIterator,F:Fn<(V::Item,)>>(v: V, f: F) -> Map<V, F> { IntoMap::map(v, f) }

impl<I:IntoIterator, F: Fn<(I::Item,)>> IntoIterator for Map<I, F> {
	type IntoIter = std::iter::Map::<I::IntoIter, F>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::map(self.iter.into_iter(), self.f) }
}
#[macro_export] macro_rules! map { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::into::IntoMap::map($crate::zip!($($args,)*), |($($params),*)| $expr) }; }

pub trait Find : IntoIterator+Sized { fn find<P:FnMut(&Self::Item)->bool>(self, predicate: P) -> Option<Self::Item> { self.into_iter().find(predicate) } }
impl<I:IntoIterator> Find for I {}

pub trait FilterMap : IntoIterator+Sized { fn filter_map<U, F:FnMut(Self::Item)->Option<U>>(self, f: F) -> std::iter::FilterMap<Self::IntoIter, F> { self.into_iter().filter_map(f) } }
impl<I:IntoIterator> FilterMap for I {}

pub trait Sum<T> { fn sum(self) -> T; }
impl<I:IntoIterator, T:std::iter::Sum<I::Item>> Sum<T> for I { fn sum(self) -> T { Iterator::sum(self.into_iter()) } }

pub trait Product<T> { fn product(self) -> T; }
impl<I:IntoIterator, T:std::iter::Product<I::Item>> Product<T> for I { fn product(self) -> T { Iterator::product(self.into_iter()) } }
