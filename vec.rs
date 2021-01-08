use super::{into, IntoIterator};

pub trait Vector<const N: usize> : IntoIterator<IntoIter:ExactSizeIterator>+Sized { //super::IntoExactSizeIterator+Sized { //IntoIterator<IntoIter:ExactSizeIterator> {
	//type Item = <Self as IntoIterator>::Item;
	fn collect(self) -> [<Self as IntoIterator>::Item; N] { super::FromExactSizeIterator::from_iter(self) }
}
//impl<V:Vector<N>, const N: usize> Vector<N> for &'t V where &'t V:IntoIterator {}
impl<T, const N: usize> Vector<N> for &[T; N] {}
impl<T, const N: usize> Vector<N> for [T; N] {}
//impl<T, const N: usize> Vector<N> for super::IntoIter<'_, T, N> {}
impl<T:Copy+'t, I:IntoIterator<Item=&'t T>+Vector<N>, const N: usize> Vector<N> for super::into::Copied<I> {} //where Self:IntoIterator<IntoIter:ExactSizeIterator> {}
impl<I:Vector<N>, F:Fn<(<I as IntoIterator>::Item,)>, const N: usize> Vector<N> for into::Map<I, F> {}
impl<A:Vector<N>, B:Vector<N>, const N: usize> Vector<N> for into::Zip<A, B> {}

/*pub trait VectorCollect<const N: usize> : Vector<N> {
	fn collect(self) -> [Self::Item; N];
}
impl<T,V:Vector<N,Item=T>+IntoIterator<Item=T,IntoIter:ExactSizeIterator>, const N: usize> VectorCollect<N> for V {
	fn collect(self) -> [Self::Item; N] { super::FromExactSizeIterator::from_iter(self.into_iter()) }
}*/

pub fn eval<T, U, const N: usize>(v: impl Vector<N>+IntoIterator<Item=T,IntoIter:ExactSizeIterator>, f: impl Fn(T)->U) -> [U; N] { Vector::collect(into::map(v, f)) }
#[macro_export] macro_rules! eval { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::vec::eval($crate::zip!($($args,)*), |($($params),*)| $expr) }; }

#[derive(Clone, Copy)] pub struct ConstRange<const N: usize>;
impl<const N: usize> IntoIterator for ConstRange<N> { type IntoIter = std::ops::Range<usize>; type Item = <Self::IntoIter as Iterator>::Item; fn into_iter(self) -> Self::IntoIter { 0..N } }
impl<const N: usize> Vector<N> for ConstRange<N> {}
#[track_caller] pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> into::Map<ConstRange<N>, F> { into::map(ConstRange, f) }
//#[track_caller] pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> [T; N] { eval(ConstRange::new(), f) }

pub trait Scale { type Output; fn scale(self, s: f64) -> Self::Output; }
impl<const N: usize> Scale for &'t [f64; N] {
	type Output = [f64; N];
	fn scale(self, s: f64) -> Self::Output { eval(self, |v| s*v) }
}

pub trait Sub<B=Self> { type Output; fn sub(self, b: B) -> Self::Output; }
impl<T: 't, const N: usize> Sub for &'t [T; N] where &'t T:std::ops::Sub {
	type Output = [<&'t T as std::ops::Sub>::Output; N];
	fn sub(self, b: &'t [T; N]) -> Self::Output { eval!(self, b; |a,b| a-b) }
}

pub trait Mul<B=Self> { type Output; fn mul(self, b: B) -> Self::Output; }
impl<T: 't, const N: usize> Mul for &'t [T; N] where &'t T:std::ops::Mul {
	type Output = [<&'t T as std::ops::Mul>::Output; N];
	fn mul(self, b: &'t [T; N]) -> Self::Output { eval!(self, b; |a,b| a*b) }
}

pub trait Dot<T, const N: usize> { type Output; fn dot(self, other: T) -> Self::Output; }
impl<A: Vector<N>+IntoIterator<Item: std::ops::Mul<<B as IntoIterator>::Item>>, B: Vector<N>, const N: usize> Dot<B, N> for A
where <<A as IntoIterator>::Item as std::ops::Mul<<B as IntoIterator>::Item>>::Output: std::iter::Sum, Self:Vector<N>, B:Vector<N> {
	type Output = <<A as IntoIterator>::Item as std::ops::Mul<<B as IntoIterator>::Item>>::Output;
	fn dot(self, b: B) -> Self::Output { super::into::Sum::sum(super::map!(self, b; |a,b| a*b)) }
}
