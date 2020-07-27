#pragma once
#include <fstream>
#include <vector>
//#include <Eigen/Dense>
//#include <Eigen/Sparse>
//#include <Eigen/SVD>

#include "../../Framework/source/StringUtilities.h"
#include "../../Framework/source/io/File.h"

namespace Jde::Math
{
	template<typename T>
	using SparseT = Eigen::SparseMatrix<T>;

	template<typename T>
	using SparseTPtr = std::unique_ptr<Eigen::SparseMatrix<T>>;
	namespace ESparse
	{
		template<typename T>
		void ForEach( const Eigen::SparseVector<T>& vector, const function<void(int,const T*)>& function );
		template<typename T>
		void ForEach( const SparseT<T>& sparse, const function<void(int,int,const T*)>& function );

		template<typename T>
		void ForEachValue( const SparseT<T>& matrix, const std::function<void(int,int,T)>& function, const std::vector<int>* pColumnIndexes=nullptr, Stopwatch* pStopwatch=nullptr );
		template<typename T>
		void ForEachValue2( const Eigen::SparseVector<T>& vector, const std::function<void(int,T&)>& function );

		//template<typename T>
		//void ForEachValue( const SparseT<T>& matrix, , const std::function<void(int,int,const T&)>& function );

		template<typename T>
		void ExportCsv( const string& fileName, const SparseT<T>& matrix, const vector<string>* pColumnNames=nullptr );
		template<typename T>
		pair<unique_ptr<VectorXi>,std::map<int,int>> GetColumnCounts( const SparseT<T>& sparse, int j, function<bool(int i,const T& value)> where );
		template<typename T>
		SparseTPtr<T> LoadCsv( const std::string& csvFileName, std::vector<string>& columnNamesToFetch, size_t maxLines=std::numeric_limits<size_t>::max(), bool notColumns=false, int chunkSize=100000 );
		template<typename T,int Cols=-1>
		unique_ptr<Matrix<T,-1,Cols>> Mean( const SparseT<T>& sparse, const int groupByIndex, const std::vector<int>& meanIndexes );
		template<typename T>
		std::pair<std::pair<SparseTPtr<T>,SparseTPtr<T>>, std::pair<SparseTPtr<T>,SparseTPtr<T>> > Split( const SparseT<T>& x, const SparseT<T>& y, double percent );
		template<typename T>
		SparseTPtr<T> Split( const SparseT<T>& x, bool odd );
		template<typename T>
		std::pair<SparseTPtr<T>, std::pair<SparseTPtr<T>,up<Eigen::SparseVector<T>>> > SplitSuccess( const SparseT<T>& x, const Eigen::SparseVector<T>& y, double percent );

		template<typename T>
		SparseTPtr<T> AddColumns3( const SparseT<T>& sparse, size_t removeColumnCount, const std::initializer_list<const SparseT<T>*> append );
		template<typename T, int Rows=-1, int Cols=-1>
		SparseTPtr<T> AddColumns( const SparseT<T>& sparse, const Eigen::Matrix<T,Rows,Cols>& append  );
		template<typename T>
		SparseTPtr<T> AddColumns( const SparseT<T>& sparse, const std::vector<std::tuple<std::function<T(T,T)>,int,T,bool>>& functions );
		template<typename T>
		SparseTPtr<T> AddColumns( const SparseT<T>& sparse, const size_t count );
		template<typename T>
		SparseTPtr<T> AddRows( const SparseT<T>& sparse1, const SparseT<T>& sparse2, bool addIndex=false );
		template<typename T>
		SparseTPtr<T> BsxFun( const SparseT<T>& matrix1, const SparseT<T>& matrix2, const std::function<T(T,T)>& binaryFunction );
		template<typename T>
		SparseTPtr<T> BsxFun( const SparseT<T>& matrix, const Eigen::Matrix<T,1,-1>& rowVector, const std::function<T(T,T)>& binaryFunction );
		template<typename T>
		SparseTPtr<T> BsxFunSubtract( const SparseT<T>& matrix, const SparseT<T>& rowVector ){ return BsxFun<T>( matrix, rowVector, [](T value1, T value2){return value1-value2;} ); }
		template<typename T>
		SparseTPtr<T> BsxFunSubtract( const SparseT<T>& matrix, const Eigen::Matrix<T,1,-1>& rowVector ){ return BsxFun<T>( matrix, rowVector, [](T value1, T value2){return value1-value2;} ); }
		template<typename T>
		SparseTPtr<T> BsxFunMultiply( const SparseT<T>& matrix1, const SparseT<T>& matrix2 ){ return BsxFun<T>( matrix1, matrix2, [](T value1, T value2){return value1*value2;} ); }
		//template<typename T>
		//SparseTPtr<T> Centered( const SparseT<T>& sparse );
		template<typename T>
		SparseTPtr<T> CreateSparseMatrix( const SparseT<T>& copyFrom );
		template<typename T>
		up<Eigen::SparseVector<T>> CreateSparseVector( const Eigen::SparseVector<T>& copyFrom );
		template<typename T>
		SparseTPtr<T> Merge( const SparseT<T>& sparse, const SparseT<T>& append  );
		template<typename T>
		std::tuple<SparseTPtr<T>,up<Eigen::SparseVector<T>>,up<Eigen::SparseVector<T>>> Normalize( const Eigen::SparseMatrix<T>& matrix, bool sample=true );
		template<typename T>
		SparseTPtr<T> Prefix( const SparseT<T>& matrix, T value );
		template<typename T>
		SparseTPtr<T> SparseRowwise( const SparseT<T>& matrix1, const Eigen::SparseVector<T>& rows, const std::function<T(T,T)>& binaryFunction );
		template<typename T, int SortCount>
		SparseTPtr<T> Sort(const SparseT<T>& matrix1, std::array<int,SortCount> columnIndexes, Stopwatch* pStopwatch=nullptr );
		template<typename T>
		up<Eigen::SparseVector<T>> AverageColumns( const SparseT<T>& matrix, bool valuesOnly/*=true*/ );
		template<typename T>
		size_t Count( const Eigen::SparseVector<T>& vector, const std::function<bool(int index,T value)> where );
		template<typename T>
		up<Eigen::SparseVector<T>> SumColumns( const SparseT<T>& matrix );

		template<typename T>
		std::pair<up<Eigen::SparseVector<T>>,up<Eigen::SparseVector<T>>> VarianceColumns( const SparseT<T>& matrix, bool sample=true, bool valuesOnly=true );
	}
#pragma region ForEach
	template<typename T>
	void ESparse::ForEach( const Eigen::SparseVector<T>& v, const function<void(int,const T*)>& function )
	{
		int i = 0;

		for( typename Eigen::SparseVector<T>::InnerIterator it(v); it; ++it, ++i )
		{
			for( ;i<it.row(); ++i )
				function( i, nullptr );
			function( i, &it.valueRef() );
		}
	}
	template<typename T>
	void ESparse::ForEach( const SparseT<T>& sparse, const function<void(int,int,const T*)>& function )
	{
		for( int j=0; j<sparse.cols(); ++j )
		{
			int i = 0;
			for( typename SparseT<T>::InnerIterator it(sparse,j); it; ++it, ++i )
			{
				for( ;i<it.row(); ++i )
					function( i, j, nullptr );
				function( i, j, &it.value() );
			}
		}
	}

#pragma endregion
#pragma region ForEachValue
	template<typename T>
	void ESparse::ForEachValue( const SparseT<T>& sparse, const std::function<void(int,int,T)>& function, const std::vector<int>* pColumnIndexes/*=nullptr*/, Stopwatch* pStopwatch )
	{
		size_t index=0;
		const size_t nonZeros = sparse.nonZeros();
		const size_t progressIndex = std::max( size_t(10000), size_t(nonZeros/10) );
		const size_t columnCount = pColumnIndexes ? pColumnIndexes->size() : sparse.outerSize();
		//clog << "ForEachValue::columnCount = " << columnCount <<  std::endl;
		for( auto columnIndex=0; columnIndex<columnCount; ++columnIndex )
		{
			const int j = pColumnIndexes ? (*pColumnIndexes)[columnIndex] : columnIndex;
			for( typename SparseT<T>::InnerIterator it(sparse,j); it; ++it )
			{
				const auto rowIndex = it.row();
				auto value = it.value();
				function( rowIndex, columnIndex, value );
				if( pStopwatch!=nullptr && ++index%progressIndex==0 )
					pStopwatch->Progress( index, nonZeros );
			}
		}
	}
	template<typename T>
	void ESparse::ForEachValue2( const Eigen::SparseVector<T>& vector, const std::function<void(int,T&)>& function )
	{
		for( typename Eigen::SparseVector<T>::InnerIterator it(vector,0); it; ++it )
			function( it.row(), it.valueRef() );
	}
#pragma endregion
#pragma region Count
	template<typename T>
	size_t ESparse::Count( const Eigen::SparseVector<T>& vector, const std::function<bool(int index,T value)> where )
	{
		size_t count = 0;
		ForEachValue2<T>( vector, [&count, &where](int index,T& value)mutable{if( where(index, value) ) ++count;} );
		return count;
	}
#pragma endregion

#pragma region AverageColumns
	template<typename T>
	up<Eigen::SparseVector<T>> ESparse::AverageColumns( const SparseT<T>& matrix, bool valuesOnly/*=true*/ )
	{
		const auto columnCount = matrix.cols();

		const auto sum = SumColumns<T>( matrix );
		auto average = new Eigen::SparseVector<T>( columnCount );
		average->reserve( columnCount );
		if( valuesOnly )
		{
			throw new Exception( "not Implemented." );
/*			Eigen::Matrix<size_t,1,-1> columnCounts( 1, columnCount );
			auto function = [&average,&columnCounts](int columnIndex, T value)mutable{ average->coeffRef(columnIndex)+=value; ++columnCounts(0,columnIndex); };
			TransformEachValue<T>( average, [&columnCounts](int columnIndex, T& value)mutable{ size_t columnCount = columnCounts(0,columnIndex); if(columnCount==0) value=0; else value/=columnCount;} );*/
		}
		else
		{
			const T rowCountF = T(matrix.rows());
			auto function = [&average,rowCountF](int columnIndex, T value)mutable{ average->coeffRef(columnIndex)=value/rowCountF; };
			ForEachValue2<T>( *sum, function );
		}
		return up<Eigen::SparseVector<T>>(average);
	}
#pragma endregion
#pragma region ExportCsv
	template<typename T>
	void ESparse::ExportCsv( const string& fileName, const SparseT<T>& matrix, const vector<string>* pColumnNames )
	{
		std::ofstream myfile;
		myfile.open( fileName );
		if( !myfile.good() )
			throw "wtf";
		if( pColumnNames )
		{
			bool first = true;
			for( const auto& columnName : *pColumnNames )
			{
				if( first )
					first = false;
				else
					myfile << ",";
				myfile << columnName;
			}
			myfile << endl;
		}
		for( int i=0; i<matrix.rows(); ++i )
		{
			myfile << matrix.coeff(i,0);
			for( int j = 1; j<matrix.cols(); ++j )
				myfile << "," << matrix.coeff( i, j );
			myfile << endl;
		}
		myfile.close();
	}
#pragma endregion
#pragma region GetColumnCounts
	template<typename T>
	pair<unique_ptr<VectorXi>,std::map<int,int>> ESparse::GetColumnCounts( const SparseT<T>& sparse, int j, function<bool(int i,const T& value)> where )
	{
		auto pColumnCounts = new VectorXi( VectorXi::Zero( sparse.cols()) );
		std::map<int,int> rowIndexes;
		auto pHint = rowIndexes.begin();
		int newRowIndex=0;
		auto findFunction = [&where, &rowIndexes,&pHint,&newRowIndex](int i, const T* pValue)mutable
		{
			if( where(i, pValue ? *pValue : T(0)) )
				pHint = rowIndexes.emplace_hint( pHint, i, newRowIndex++ );
		};
		ESparse::ForEach<T>( sparse.col(j), findFunction );

		auto countFunction = [&sparse,&rowIndexes,&pColumnCounts,&pHint,&newRowIndex](int i, int j, T value)mutable
		{
			if( rowIndexes.find(i)!=rowIndexes.end() )
			{
				//if( j==15 )
				//clog << "[" << i << ", " << j << "]=" << value << std::endl;
				++pColumnCounts->coeffRef(j);
			}
		};
		ForEachValue<T>( sparse, countFunction );

		return make_pair( up<VectorXi>(pColumnCounts), rowIndexes );
	}

#pragma endregion
#pragma region LoadCsv
#define var const auto
	template<typename T>
	SparseTPtr<T> ESparse::LoadCsv( const std::string& csvFileName, std::vector<string>& columnNamesToFetch, size_t maxLines, bool notColumns/*=false*/, int chunkSize/*=100000*/ )
	{
		Stopwatch sw( csvFileName );
		std::vector<std::string> columnNames;
		const bool allColumns = columnNamesToFetch.size()==0;

		std::set<size_t> columnIndexes;
		if( !allColumns )
		{
			auto columnNameFunction = [&columnNamesToFetch,&columnIndexes,&columnNames,&notColumns](std::string line)
			{
				var tokens = StringUtilities::Split<char>(line);
				int iToken=0;
				for( const auto& token : tokens )
				{
					if( (!notColumns && std::find( columnNamesToFetch.begin(), columnNamesToFetch.end(), token)!=columnNamesToFetch.end())
						|| (notColumns && std::find( columnNamesToFetch.begin(), columnNamesToFetch.end(), token)==columnNamesToFetch.end()) )
					{
						columnNames.push_back( token );
						columnIndexes.insert( iToken );
					}
					++iToken;
				}
			};
			IO::File::ForEachLine<char>( csvFileName, columnNameFunction, 1 );
		}
		vector<vector<double>> columnValues(columnNames.size());
		vector<vector<int>> rowIndexes(columnNames.size());
		for( int columnIndex=0; columnIndex<columnNames.size(); ++columnIndex )
		{
			columnValues[columnIndex].reserve( chunkSize );
			rowIndexes[columnIndex].reserve( chunkSize );
		}

		auto getValues = [&sw,&columnValues,&rowIndexes,&chunkSize]( const std::vector<double>& tokens, size_t lineIndex )mutable
		{
			if( tokens.size()!=columnValues.size() )
				THROW( Exception(fmt::format( "Column counts don't add up for line '{}' actual:  '{}' expected:  '{}'", lineIndex, tokens.size(), columnValues.size()) ) );

			auto columnIndex = 0;
			for( auto pToken = tokens.begin(); pToken!=tokens.end(); ++columnIndex, ++pToken )
			{
				const double& value = *pToken;
				auto& values = columnValues[columnIndex];
				auto& rows = rowIndexes[columnIndex];
				if( values.capacity()==values.size() )
				{
					values.reserve( values.size()+chunkSize );
					rows.reserve( rows.size()+chunkSize );
				}
				values.push_back( value );
				rows.push_back( int(lineIndex-1) );
			}
			if( lineIndex%100000==0 )
				sw.Progress( lineIndex );
		};
		size_t lineCount = IO::File::ForEachLine4( csvFileName.c_str(), getValues, columnIndexes, maxLines, 1, 1073741824, 1500, &sw );
		sw.Finish();

		Stopwatch sw2( "sparse" );
		VectorXi reserve( columnValues.size() );
		int valueCount = 0;
		for( auto columnIndex=0; columnIndex<columnValues.size(); ++columnIndex )
		{
			const int count = int(columnValues[columnIndex].size());
			valueCount+=count;
			reserve(columnIndex) = count;
		}

		auto pResult = new SparseT<T>( int(lineCount), int(columnValues.size()) );
		pResult->reserve( reserve );

		for( int columnIndex=0; columnIndex<columnValues.size(); ++columnIndex )
		{
			const auto& values = columnValues[columnIndex];
			const auto& rows = rowIndexes[columnIndex];
			for( int valueIndex=0; valueIndex<values.size(); ++valueIndex )
			{
				pResult->coeffRef( rows[valueIndex], columnIndex ) = T(values[valueIndex]);
			}
		}
		return up<SparseT<T>>(pResult);
	}
#pragma endregion
#pragma region Mean
	template<typename T, int Cols>
	unique_ptr<Matrix<T,-1,Cols>> ESparse::Mean( const SparseT<T>& sparse, const int groupByIndex, const std::vector<int>& meanIndexes )
	{
		map<T,pair<size_t,vector<T>>> countSums;
		auto pValueSums = countSums.begin();
		auto function = [&countSums,&pValueSums,&meanIndexes,&sparse]( int i, int j, T value )
		{
			pValueSums = countSums.emplace_hint( pValueSums, value, make_pair(0, vector<T>(meanIndexes.size())) );
			pair<size_t,vector<T>>& countSums = pValueSums->second;
			++countSums.first;
			vector<T>& sums = countSums.second;

			for( int sumIndex=0; sumIndex< meanIndexes.size(); ++sumIndex )
				sums[sumIndex]+=sparse.coeff( i, meanIndexes[sumIndex] );
		};
		auto values2 = vector<int>{groupByIndex};
		ForEachValue<T>( sparse, function, &values2 );
		auto pResult = new Matrix<T,-1,Cols>( countSums.size(), meanIndexes.size()+1 );
		int i=0;
		for( const auto& values : countSums )
		{
			pResult->coeffRef( i, 0 ) = values.first;
			int j=0;
			for( const T& sum : values.second.second )
				pResult->coeffRef(i,++j) = sum/T(values.second.first);
			++i;
		}
		return unique_ptr<Matrix<T,-1,Cols>>( pResult );
	}
#pragma endregion
#pragma region Split
	template<typename T>
	SparseTPtr<T> ESparse::Split( const SparseT<T>& sparse, bool odd )
	{
		function<bool(int)> oddWhere = [](int i){ return i%2==0; };
		function<bool(int)> evenWhere = [](int i){ return i%2==1; };
		auto where = odd ? oddWhere : evenWhere;
		//int newRowIndex=0;
		//map<int,int> newRowIndexes;
		//auto pHint = newRowIndexes.begin();
		function<bool(int,const T&)> function = [&where](int i, const T&)mutable
		{
			return where( i );
			//bool include = where(i);
			//if( include )
			//	newRowIndexes.emplace_hint( pHint, i, newRowIndex++ );
			//return include;
		};
		auto columnCountsIndexes = GetColumnCounts( sparse, 0, function );
		auto newRowIndexes = columnCountsIndexes.second;
		auto pResult = new SparseT<T>( int(newRowIndexes.size()), sparse.cols() );
		pResult->reserve( *columnCountsIndexes.first );
		//int newRowIndex=0;
		//function<void(int,int,T)>
		auto insertProc = [&where,&newRowIndexes,&pResult](int i, int j, T value)
		{
			if( where(i) )
				pResult->insert( newRowIndexes[i], j )=value;
		};
		ForEachValue<T>( sparse, insertProc );

		return SparseTPtr<T>( pResult );
	}

	template<typename T>
	std::pair<std::pair<SparseTPtr<T>,SparseTPtr<T>>, std::pair<SparseTPtr<T>,SparseTPtr<T>> > ESparse::Split( const SparseT<T>& x, const SparseT<T>& y, double percent )
	{
		const int rowCount = x.rows();
		const int xColumnCount = x.cols();
		const int yColumnCount = y.cols();
		const up<Matrix<int,-1,1>> pRandomIndexes = EMatrix::RandomIndexes<-1,int>( rowCount );
		const int x1Count = int(std::round(double(rowCount)*percent));
		Eigen::VectorXd column1Counts = Eigen::VectorXd::Zero(xColumnCount), column2Counts = Eigen::VectorXd::Zero(xColumnCount);
		Eigen::VectorXd column1YCounts = Eigen::VectorXd::Zero(yColumnCount), column2YCounts = Eigen::VectorXd::Zero(yColumnCount);
		//VectorXi* pColumnCounts = nullptr;
		for( int randomIndex=0; randomIndex<rowCount; ++randomIndex )
		{
			const bool isMatrix1 = randomIndex<x1Count;
			auto pColumnCounts = isMatrix1 ? &column1Counts : &column2Counts;
			const int rowIndex = int(pRandomIndexes->coeff(randomIndex));
			for( int columnIndex = 0; columnIndex<xColumnCount; ++columnIndex )
			{
				const T value = x.coeff(rowIndex, columnIndex);
				if( value!=T(0) )
					pColumnCounts->coeffRef(columnIndex)++;
			}
			auto pYColumnCounts = isMatrix1 ? &column1YCounts : &column2YCounts;
			for( int columnIndex = 0; columnIndex<yColumnCount; ++columnIndex )
			{
				const T value = y.coeff( rowIndex, columnIndex );
				if( value!=T(0) )
					pYColumnCounts->coeffRef(columnIndex)++;
			}
		}

		SparseT<T>* pX1 = new SparseT<T>( x1Count, xColumnCount ), *pX2 = new SparseT<T>( rowCount-x1Count, xColumnCount );
		pX1->reserve( column1Counts );
		pX2->reserve( column2Counts );
		SparseT<T>* pY1 = new SparseT<T>( x1Count, yColumnCount ), *pY2 = new SparseT<T>( rowCount-x1Count, yColumnCount );
		pY1->reserve( column1YCounts );
		pY2->reserve( column2YCounts );
		int x1Index=0, x2Index=0, y1Index=0, y2Index=0;
		for( int randomIndex=0; randomIndex<rowCount; ++randomIndex )
		{
			const int existingRowIndex = int( pRandomIndexes->coeff(randomIndex) );
			const bool isMatrix1 = randomIndex<x1Count;
			SparseT<T>* pX = isMatrix1 ? pX1 : pX2;
			int* pNewIndex = isMatrix1 ? &x1Index : &x2Index;
			for( int columnIndex = 0; columnIndex<xColumnCount; ++columnIndex )
			{
				const T value = x.coeff(existingRowIndex, columnIndex);
				if( value!=T(0) )
					pX->coeffRef(*pNewIndex, columnIndex) = value;
			}

			SparseT<T>* pY = isMatrix1 ? pY1 : pY2;
			int* pYNewIndex = isMatrix1 ? &y1Index : &y2Index;
			bool haveValue = false;
			for( int columnIndex = 0; columnIndex<yColumnCount; ++columnIndex )
			{
				const T value = y.coeff(existingRowIndex, columnIndex);
				if( value!=T(0) )
				{
					haveValue = true;
					pY->coeffRef(*pNewIndex, columnIndex) = value;
				}
			}
			//if( haveValue )
			(*pYNewIndex)++;

			(*pNewIndex)++;
		}
		return std::make_pair( std::make_pair(SparseTPtr<T>(pX1),SparseTPtr<T>(pY1)), std::make_pair(SparseTPtr<T>(pX2),SparseTPtr<T>(pY2)) );
	}
	template<typename T>
	std::pair<SparseTPtr<T>, std::pair<SparseTPtr<T>,up<Eigen::SparseVector<T>>> > ESparse::SplitSuccess( const SparseT<T>& x, const Eigen::SparseVector<T>& y, double percent )
	{
		const int rowCount = x.rows();
		const int xColumnCount = x.cols();
		const int successCount = int( EMatrix::Count( y, 0.0f ) );
		const int trainingCount = int( std::round(T(rowCount)*percent) );
		const up<Matrix<int,-1,1>> pRandomIndexes = EMatrix::RandomIndexes<-1,int>( rowCount );

		Eigen::VectorXd column1Counts = Eigen::VectorXd::Zero(xColumnCount), column2Counts = Eigen::VectorXd::Zero(xColumnCount);
		int column2YCounts = rowCount-successCount;
		//VectorXi* pColumnCounts = nullptr;
		size_t trainIndex = 0;
		for( int randomIndex=0; randomIndex<rowCount; ++randomIndex )
		{
			const int rowIndex = pRandomIndexes->coeff(randomIndex);
			const T yValue = y.coeff( rowIndex );
			const bool success = yValue==0.0f;
			const bool isTraining = success && trainIndex<trainingCount;
			auto pColumnCounts = isTraining ? &column1Counts : &column2Counts;
			for( int columnIndex = 0; columnIndex<xColumnCount; ++columnIndex )
			{
				const T value = x.coeff(rowIndex, columnIndex);
				if( value!=T(0) )
					pColumnCounts->coeffRef(columnIndex)++;
			}
			if( isTraining )
				++trainIndex;
		}
		SparseT<T>* pX1 = new SparseT<T>( trainingCount, xColumnCount ), *pX2 = new SparseT<T>( rowCount-trainingCount, xColumnCount );
		pX1->reserve( column1Counts );
		pX2->reserve( column2Counts );
		Eigen::SparseVector<T>* pY2 = new Eigen::SparseVector<T>( rowCount-trainingCount );
		pY2->reserve( column2YCounts );
		int x1Index=0, x2Index=0, y2Index=0;
		trainIndex = 0;
		for( int randomIndex=0; randomIndex<rowCount; ++randomIndex )
		{
			const int existingRowIndex = pRandomIndexes->coeff(randomIndex);
			const T yValue = y.coeff( existingRowIndex );
			const bool success = yValue==0.0f;
			const bool isTraining = success && trainIndex<trainingCount;
			SparseT<T>* pX = isTraining ? pX1 : pX2;
			int* pNewIndex = isTraining ? &x1Index : &x2Index;
			for( int columnIndex = 0; columnIndex<xColumnCount; ++columnIndex )
			{
				const T value = x.coeff(existingRowIndex, columnIndex);
				if( value!=T(0) )
					pX->coeffRef(*pNewIndex, columnIndex) = value;
			}
			if( !isTraining )
			{
				if( yValue!=T(0) )
					pY2->coeffRef(y2Index) = yValue;
				++y2Index;
			}
			(*pNewIndex)++;
			if( isTraining )
				++trainIndex;
		}
		return std::make_pair( SparseTPtr<T>(pX1), std::make_pair(SparseTPtr<T>(pX2),up<Eigen::SparseVector<T>>(pY2)) );
	}
#pragma region AddColumns
	template<typename T>
	SparseTPtr<T> ESparse::AddColumns3( const SparseT<T>& sparse, size_t removeColumnCount, const std::initializer_list<const SparseT<T>*> additions )
	{
		auto columnCount = sparse.cols();
		up<Eigen::VectorXi> pColumnCounts = GetColumnCounts( sparse );
		for( const auto& pAppend : additions )
		{
			columnCount+=pAppend->cols()-int(removeColumnCount);
			up<Eigen::VectorXi> pNewColumnCounts = up<Eigen::VectorXi>( new Eigen::VectorXi(columnCount) );
			*pNewColumnCounts << *pColumnCounts, GetColumnCounts( *pAppend )->bottomRows( pAppend->cols()-removeColumnCount );

			pColumnCounts = std::move(pNewColumnCounts);
		}


		auto pResult = new SparseT<T>( sparse.rows(), (int)columnCount );
		pResult->reserve( *pColumnCounts );
		ESparse::ForEachValue<T>( sparse, [&pResult](int i, int j, T value)mutable{pResult->insert(i,j)=value;} );
		int startColumn = sparse.cols();
		for( const auto& pAppend : additions )
		{
			ESparse::ForEachValue<T>( *pAppend, [&pResult,&startColumn,&removeColumnCount](size_t i, size_t j, T value)mutable{if( j<removeColumnCount ) return; pResult->insert(int(i),int(startColumn+j-removeColumnCount))=value;} );
			startColumn+=pAppend->cols()-int(removeColumnCount);
		}

		return SparseTPtr<T>(pResult);
	}

	template<typename T, int Rows, int Cols>
	SparseTPtr<T> ESparse::AddColumns( const SparseT<T>& sparse, const Eigen::Matrix<T,Rows,Cols>& append  )
	{
		const auto sparseCols = sparse.cols();
		const auto columnCount = sparseCols+append.cols();
		Eigen::VectorXi* pColumnCounts = nullptr;
		if( sparse.innerNonZeroPtr()!=nullptr )
		{
			pColumnCounts = new Eigen::VectorXi( Eigen::VectorXi::Zero(columnCount) );
			for( int columnIndex = 0; columnIndex<columnCount-1; ++columnIndex )
				pColumnCounts->coeffRef(columnIndex) = sparse.innerNonZeroPtr()[columnIndex];
			for( int j=sparseCols; j<columnCount; ++j )
				pColumnCounts->coeffRef( j ) = sparse.rows();
		}
		else
			throw new Exception( "not Implemented." );
		SparseT<T>* pResult = new SparseT<T>( sparse.rows(), (int)columnCount );
		if( pColumnCounts!=nullptr )
			pResult->reserve( *pColumnCounts );
		ESparse::ForEachValue<T>( sparse, [&pResult](int i, int j, T value)mutable{pResult->coeffRef(i,j)=value;} );
		ForEach<T,Rows,Cols>( append, [&pResult,&sparseCols](size_t i, size_t j, T value)mutable{pResult->coeffRef(int(i),int(sparseCols+j))=value;} );

		return SparseTPtr<T>(pResult);
	}
	template<typename T>
	SparseTPtr<T> ESparse::AddColumns( const SparseT<T>& sparse, const std::vector<std::tuple<std::function<T(T,T)>,int,T,bool>>& functions )
	{
		const auto sparseCols = sparse.cols();
		const auto functionCount = functions.size();
		const auto columnCount = sparseCols + functionCount;
		std::map<size_t,std::vector<T>> appendColumnValues;
		typename std::map<size_t,std::vector<T>>::iterator pRow = appendColumnValues.begin();
		int functionIndex = 0;
		Eigen::VectorXi columnCounts = Eigen::VectorXi::Zero( columnCount );
		for( const auto& item : functions )
		{
			const auto& function = std::get<0>(item);
			const auto& j = std::get<1>(item);
			const auto& naValue = std::get<2>(item);
			const auto& reverse = std::get<3>(item);

			//size_t index=0;
			T previous = std::numeric_limits<T>::max();
			int previousIndex = -1;

			pRow = appendColumnValues.emplace_hint( pRow, reverse ? sparse.rows()-1 : 0, functionCount );
			pRow->second[functionIndex] = naValue;
			for( typename Eigen::SparseMatrix<T>::InnerIterator it(sparse,j); it; ++it )
			{
				const auto rowIndex = it.row();
				const T value = it.value();
				const size_t insertIndex = reverse ? rowIndex-1 : rowIndex;
				if( rowIndex!=previousIndex+1 )//1 or more 0 cells
				{
					auto result = previous!=T(0) && previousIndex!=-1 ? function( previous, T(0) ) : T(0);
					if( result!=T(0) )
					{
						pRow = appendColumnValues.emplace_hint( pRow, insertIndex, functionCount );
						pRow->second[functionIndex] = result;
						++columnCounts(sparseCols+functionIndex);
					}
					previous = 0;
				}
				if( rowIndex!=0 )
				{
					auto result = function( previous, value );
					if( result!=T(0) && insertIndex<sparse.rows() )
					{
						pRow = appendColumnValues.emplace_hint( pRow, insertIndex, functionCount );
						pRow->second[functionIndex] = result;
						++columnCounts(sparseCols+functionIndex);
					}
				}
				previous = value;
				previousIndex = rowIndex;
			}
			++functionIndex;
		}

		if( sparse.innerNonZeroPtr()==nullptr )
			throw new Exception( "not Implemented." );

		for( int columnIndex = 0; columnIndex<sparseCols; ++columnIndex )
			columnCounts(columnIndex) = sparse.innerNonZeroPtr()[columnIndex];

		SparseT<T>* pResult = new SparseT<T>( sparse.rows(), (int)columnCount );
		pResult->reserve( columnCounts );
		ESparse::ForEachValue<T>( sparse, [&pResult](int i, int j, T value)mutable{pResult->coeffRef(i,j)=value;} );
		for( const auto& append : appendColumnValues )
		{
			const size_t i = append.first;
			int j=0;
			for( const auto& value : append.second )
			{
				if( value!=T(0) )
					pResult->coeffRef( int(i), int(sparse.cols()+j) ) = value;
				++j;
			}
		}

		return SparseTPtr<T>(pResult);
	}

	template<typename T>
	SparseTPtr<T> ESparse::AddColumns( const SparseT<T>& sparse, const size_t count )
	{
		auto pColumnCounts = GetColumnCounts( sparse );
		pColumnCounts->conservativeResize( pColumnCounts->rows()+1, Eigen::NoChange );
		pColumnCounts->coeffRef( sparse.cols(), 0 ) = 0;
		auto pResult = new SparseT<T>( sparse.rows(), sparse.cols()+1 );
		pResult->reserve( *pColumnCounts );
		ESparse::ForEachValue<T>( sparse, [&pResult](int i, int j, T value)mutable{pResult->coeffRef(i,j)=value; } );

		return SparseTPtr<T>(pResult);
	}
#pragma endregion
#pragma region AddRows
	template<typename T>
	SparseTPtr<T> ESparse::AddRows( const SparseT<T>& sparse1, const SparseT<T>& sparse2, bool addIndex/*=false*/ )
	{
		//const auto sparseCols = sparse.cols();
		const auto columnCount = max( sparse1.cols(), sparse2.cols() ) + (addIndex ? 1 : 0);
		//clog << "columnCount:  " << columnCount << ".  max(" <<max( sparse1.cols(), sparse2.cols() ) << ")  sparse1.cols():  " << sparse1.cols() <<".  sparse2.cols():  " << sparse2.cols() <<".  addIndex:  " << (addIndex ? 1 : 0) << std::endl;

		if( sparse1.innerNonZeroPtr()==nullptr || sparse2.innerNonZeroPtr()==nullptr )
			throw new Exception( "not Implemented." );
		Eigen::VectorXi columnCounts = Eigen::VectorXi::Zero( columnCount );
		for( int columnIndex = 0; columnIndex<sparse1.cols(); ++columnIndex )
			columnCounts.coeffRef(columnIndex) = sparse1.innerNonZeroPtr()[columnIndex];
		for( int columnIndex = 0; columnIndex<sparse2.cols(); ++columnIndex )
			columnCounts.coeffRef(columnIndex) += sparse2.innerNonZeroPtr()[columnIndex];

		const auto sparse1Rows = sparse1.rows();
		const auto totalRows = sparse1Rows + sparse2.rows();
		const int lastColumnIndex = columnCount-1;
		if( addIndex )
			columnCounts(lastColumnIndex) = totalRows;

		auto pResult = new SparseT<T>( sparse1.rows()+sparse2.rows(), columnCount );
		pResult->reserve( columnCounts );
		ESparse::ForEachValue<T>( sparse1, [&pResult](int i, int j, T value)mutable{pResult->coeffRef(i,j)=value;} );
		ESparse::ForEachValue<T>( sparse2, [&pResult,sparse1Rows](int i, int j, T value)mutable{pResult->coeffRef(i+sparse1Rows,j)=value;} );
		if( addIndex )
		{
			for( int i=0; i<totalRows; ++i )
				pResult->coeffRef(i,lastColumnIndex) = T(i);
		}

		return SparseTPtr<T>(pResult);
	}
#pragma endregion
#pragma region Centered
/*	template<typename T>
	SparseTPtr<T> ESparse::Centered( const SparseT<T>& sparse  )
	{
		SparseTPtr<T> pCentered = CreateSparseMatrix( sparse );
		auto function = [&pAverage, &pCentered](int rowIndex, int columnIndex, T value ){ pCentered->coeffRef(rowIndex,columnIndex) = value-pAverage->coeff(rowIndex, columnIndex); };
		ForEachValue( sparse, function );
		return pCentered;
	}*/
#pragma endregion
	template<typename T>
	SparseTPtr<T> ESparse::BsxFun( const SparseT<T>& matrix1, const SparseT<T>& matrix2, const std::function<T(T,T)>& binaryFunction )
	{
		SparseTPtr<T> result = CreateSparseMatrix(matrix1);
		const bool isRowVector = matrix2.rows()==1;
		std::function<void(int,int,T)> cellFunction = [isRowVector,&result,&matrix2, &binaryFunction](int rowIndex,int columnIndex,T value)mutable
		{
			const int matrix2RowIndex = isRowVector ? 0 : rowIndex;
			const T cellValue = binaryFunction( value, matrix2.coeff(matrix2RowIndex, columnIndex) );
			result->coeffRef(rowIndex,columnIndex) = cellValue;
		};
		ESparse::ForEachValue<T>( matrix1, cellFunction );
		return result;
	}
	template<typename T>
	SparseTPtr<T> ESparse::BsxFun( const SparseT<T>& matrix, const Eigen::Matrix<T,1,-1>& rowVector, const std::function<T(T,T)>& binaryFunction )
	{
		SparseTPtr<T> result = CreateSparseMatrix(matrix);
		std::function<void(int,int,T)> cellFunction = [&result,&rowVector, &binaryFunction](int rowIndex,int columnIndex,T value)mutable
		{
			const T cellValue = binaryFunction( value, rowVector.coeff(0, columnIndex) );
			result->coeffRef(rowIndex,columnIndex) = cellValue;
		};
		ESparse::ForEachValue<T>( matrix, cellFunction );
		return result;
	}
#pragma region Merge
	template<typename T>
	SparseTPtr<T> ESparse::Merge( const SparseT<T>& sparse, const SparseT<T>& append  )
	{
		std::unordered_map<T,int> valueNewIndexes;//use row indexes instead of column ids
		auto pHint = valueNewIndexes.begin();
		constexpr int columnIndexJoin = 0;
		const auto sparseColumnCount = sparse.cols();
		ESparse::ForEach<T>( sparse.col(columnIndexJoin), [&pHint,&valueNewIndexes](int i, const T* pValue)mutable{pHint = valueNewIndexes.insert( pHint, make_pair(pValue ? *pValue : T(0),i) );} );
		auto pColumnCounts1 = GetColumnCounts( sparse );

		auto appendWhere = [&valueNewIndexes](int i, const T& value){return valueNewIndexes.find(value)!=valueNewIndexes.end();};
		auto pColumnCounts2 = std::move( ESparse::GetColumnCounts<T>(append, columnIndexJoin, appendWhere).first );
		const auto appendColumnCount = append.cols();
		Eigen::VectorXi columnCounts( sparseColumnCount+appendColumnCount-1 );
		columnCounts << *pColumnCounts1, pColumnCounts2->bottomRows( pColumnCounts2->rows()-1 );

		auto pResult = new Eigen::SparseMatrix<T>( sparse.rows(), sparseColumnCount+appendColumnCount-1 );
		pResult->reserve( columnCounts );
		ESparse::ForEachValue<T>( sparse, [&pResult](int i, int j, T value){pResult->insert(i,j) = value; } );
		std::map<int,int> newOldIndexes;
		auto pHint2 = newOldIndexes.begin();
		auto insertFunction = [&pResult, &sparseColumnCount, &newOldIndexes, &valueNewIndexes, &pHint2]( int appendIndex, T& cellValue )mutable
		{
			auto pValueNewIndex = valueNewIndexes.find( cellValue );
			if( pValueNewIndex!=valueNewIndexes.end() )
				pHint2 = newOldIndexes.emplace_hint( pHint2, pValueNewIndex->second, appendIndex );
		};
		ForEachValue2<T>( append.col(columnIndexJoin), insertFunction );

		for( const auto& newOldIndex : newOldIndexes )
		{
			for( int j=1; j<appendColumnCount; ++j )
				pResult->insert(newOldIndex.first, j+sparseColumnCount-1) = append.coeff( newOldIndex.second, j );
		}
		return SparseTPtr<T>( pResult );
		//std::unordered_map<int,int> oldNewIndexes;
		/*		auto insertFunction = [&pResult, &sparseColumnCount, &oldNewIndexes, &valueNewIndexes]( int oldRowIndex, int j, T cellValue )mutable
		{
		if( j==0 )
		{
		auto pValueNewIndex = valueNewIndexes.find( cellValue );
		if( pValueNewIndex!=valueNewIndexes.end() )
		oldNewIndexes.emplace( oldRowIndex, pValueNewIndex->second );
		return;
		}
		//if( j+sparseColumnCount==983 )
		//	clog << "[" << oldRowIndex << ", " << j << "]=" << cellValue << std::endl;
		auto pOldNewIndex = oldNewIndexes.find(oldRowIndex);
		if( pOldNewIndex!=oldNewIndexes.end() )
		pResult->insert(pOldNewIndex->second,j+sparseColumnCount-1) = cellValue;
		};
		ESparse::ForEachValue<T>( append, insertFunction );

		sparse

		return SparseTPtr<T>( pResult );*/
	}
#pragma endregion
#pragma region Normalize
	template<typename T>
	std::tuple<SparseTPtr<T>,up<Eigen::SparseVector<T>>,up<Eigen::SparseVector<T>>> ESparse::Normalize( const Eigen::SparseMatrix<T>& matrix, bool sample/*=true*/ )
	{
		Stopwatch sw( "Normalize" );
		auto stdDevAverage = EMatrix::StandardDeviationColumns<T>( matrix, sample );
		auto average = std::move(stdDevAverage.second);
		auto standardDeviation = std::move(stdDevAverage.first);

		auto normalization = CreateSparseMatrix( matrix );
		const std::function<void(int,int,T)> function = [&normalization, &average, &standardDeviation ](int rowIndex, int columnIndex, T value)mutable
		{
			normalization->coeffRef(rowIndex, columnIndex)=(value-average->coeff(columnIndex))/standardDeviation->coeff(columnIndex);
		};
		ESparse::ForEachValue<T>( matrix, function, nullptr, &sw );

		return std::make_tuple( std::move(normalization), std::move(average), std::move(standardDeviation) );
	}
#pragma endregion
#pragma region Prefix
	template<typename T>
	SparseTPtr<T> ESparse::Prefix( const SparseT<T>& matrix, T value )
	{
		Stopwatch sw( "Prefix" );
		const auto rowCount = matrix.rows();
		const auto columnCount = matrix.cols();

		SparseT<T>* pResult = new SparseT<T>( rowCount, columnCount+1 );
		const int additionalRoom = value==0.0 ? 0 : rowCount;
		if( matrix.innerNonZeroPtr()!=nullptr )
		{
			Eigen::VectorXi columnCounts( columnCount+1 );
			columnCounts(0) = additionalRoom;
			for( int columnIndex = 0; columnIndex<columnCount; ++columnIndex )
				columnCounts(columnIndex+1) = matrix.innerNonZeroPtr()[columnIndex];
			pResult->reserve( columnCounts );
		}
		else
			pResult->data().resize( matrix.data().size()+additionalRoom );


		for( int rowIndex = 0; rowIndex<rowCount; ++rowIndex )
			pResult->insert(rowIndex,0) = value;
		const auto function = [pResult](int rowIndex, int columnIndex, T value)mutable{ pResult->insert(rowIndex, columnIndex+1)=value; };
		ESparse::ForEachValue<T>( matrix, function, nullptr, &sw );
		return SparseTPtr<T>( pResult );
	}
#pragma endregion
#pragma region SparseRowwise
	template<typename T>
	SparseTPtr<T> ESparse::SparseRowwise( const SparseT<T>& matrix1, const Eigen::SparseVector<T>& rows, const std::function<T(T,T)>& binaryFunction )
	{
		//auto result2 = matrix1;
		auto result = CreateSparseMatrix( matrix1 );
		function<void(int,int,T)> function = [&result, &rows,&binaryFunction](int rowIndex, int columnIndex, T value)mutable
		{
			result->coeffRef(rowIndex, columnIndex) = binaryFunction(value, rows.coeff(columnIndex));
		};
		ESparse::ForEachValue( matrix1, function );
		return result;
	}
#pragma endregion
#pragma region Sort
	template<typename T, int SortCount>
	SparseTPtr<T> ESparse::Sort(const SparseT<T>& sparse, std::array<int,SortCount> columnIndexes, Stopwatch* pStopwatch )
	{
		Stopwatch createArrays( pStopwatch, "createArrays" );
		std::vector<std::array<T,SortCount+1>> values( sparse.rows() );
		for( int j=0; j<SortCount; ++j )
		{
			for( typename Eigen::SparseMatrix<T>::InnerIterator it(sparse,columnIndexes[j]); it; ++it )
			{
				const auto rowIndex = it.row();
				values[rowIndex][j] = it.value();
			}
		}
		createArrays.Finish();
		Stopwatch sort( pStopwatch, "sort" );
		std::set<std::array<T,SortCount+1>> sortedValues;
		for( int i=0; i<values.size(); ++i )
		{
			auto& arry = values[i];
			arry[SortCount] = T(i);
			sortedValues.emplace( arry );
		}
		Stopwatch copy( pStopwatch, "Copy" );
		auto pResult = CreateSparseMatrix( sparse );
		int i=0;
		for( const auto& row : sortedValues )
		{
			const int oldRowIndex = int(row[SortCount]);
			for( auto columnIndex=0; columnIndex<sparse.cols(); ++columnIndex )
			{
				pResult->coeffRef(i, columnIndex) = sparse.coeff(oldRowIndex, columnIndex);
			}
			++i;
		}
		return pResult;
	}
#pragma endregion
#pragma region SumColumns
	template<typename T>
	up<Eigen::SparseVector<T>> ESparse::SumColumns( const SparseT<T>& matrix )
	{
		const auto columnCount = matrix.cols();
		auto result = new Eigen::SparseVector<T>( columnCount );//::Zero( matrix.rows() );
		result->reserve( columnCount );
		auto sumFunction = [result](int,int columnIndex,T value)mutable
		{
			result->coeffRef(columnIndex) += value;
		};
		ESparse::ForEachValue<T>( matrix, sumFunction );
		return up<Eigen::SparseVector<T>>( result );
	}
#pragma endregion
#pragma region CreateSparse
	template<typename T>
	SparseTPtr<T> ESparse::CreateSparseMatrix( const SparseT<T>& copyFrom )
	{
		auto columnCounts = GetColumnCounts(copyFrom);
		SparseT<T>* pResult;
		if( columnCounts==nullptr )
			pResult = new SparseT<T>( copyFrom );
		else
		{
			pResult = new SparseT<T>( copyFrom.rows(), copyFrom.cols() );
			pResult->reserve( *columnCounts );
		}
		//pResult->data().resize( copyFrom.data().size() );
		//pResult->resize( copyFrom.size() );

		return SparseTPtr<T>(pResult);
	}
	template<typename T>
	up<Eigen::SparseVector<T>> ESparse::CreateSparseVector( const Eigen::SparseVector<T>& copyFrom )
	{

		auto pResult = new Eigen::SparseVector<T>();
		const int count = copyFrom.nonZeros();
		pResult->reserve( count );
		pResult->resize( count );
		return up<Eigen::SparseVector<T>>(pResult);
	}
#pragma endregion
#pragma region VarianceColumns
	template<typename T>
	std::pair<up<Eigen::SparseVector<T>>,up<Eigen::SparseVector<T>>> ESparse::VarianceColumns( const SparseT<T>& matrix, bool sample/*=true*/, bool valuesOnly/*=true*/ )
	{
		const auto rowCount = matrix.rows();
		const auto columnCount = matrix.cols();


		auto pVariance = new Eigen::SparseVector<T>( columnCount ); pVariance->reserve( columnCount );
		auto pAverages = up<Eigen::SparseVector<T>>(nullptr);
		if( valuesOnly )
		{
			pAverages = up<Eigen::SparseVector<T>>( new Eigen::SparseVector<T>( columnCount ) );
			pAverages->reserve( columnCount );
			for( int columnIndex=0; columnIndex<columnCount; ++columnIndex )
			{
				auto valueIndexes = EMatrix::GetValues( matrix, columnIndex );
				auto pColumnValues = std::move(valueIndexes.first);
				if( pColumnValues->rows()==0 )
					pVariance->coeffRef(columnIndex) =  pAverages->coeffRef(columnIndex) = 0;
				else
				{
					//clog << "[" << columnIndex << "]=" << pColumnValues->rows() << endl;
					auto average = AverageColumns( *pColumnValues );
					auto variance = VarianceColumns( *pColumnValues, sample, &average );
					pVariance->coeffRef(columnIndex) = variance->coeff(0);
					pAverages->coeffRef(columnIndex) = average.coeff(0);
				}
			}
		}
		else
		{
			pAverages = AverageColumns(matrix, valuesOnly); //matrix.colwise().mean();
			std::function<T(T,T)> func = [](T cell, T average)
			{
				return cell-average;
			};
			auto centered = SparseRowwise<T>( matrix, *pAverages, func );
			const T denominator = T(sample ? rowCount-1 : rowCount);
			Eigen::SparseMatrix<T> cov = ( centered->adjoint() * (*centered) ) / denominator;
			centered.reset();

			for( int colIndex=0; colIndex < columnCount; ++colIndex )
				pVariance->insert(colIndex) = cov.coeff(colIndex,colIndex);
		}
		return std::make_pair( up<Eigen::SparseVector<T>>(pVariance), std::move(pAverages) );
	}
#pragma endregion
#undef var
}